//! Private Information Retrieval client integration for all four Tachyon
//! payment-protocol products: `registry`, `first-contact`, `memo`, and `tachygram`.

use anyhow::{Context, Result, ensure};
use reqwest::Client as HttpClient;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Duration};
use tachyon_pir_cli::{Client as ProductClient, LookupOutcome, MapHead, MapView, ScopeSelector};
use tachyon_pir_protocol::{
    Address, Challenge, Entry, Product, RegisterRequest, Registration, ScopeState,
    registry::{CHALLENGE_ROUTE, REGISTER_ROUTE, solve_pow},
};
use udon::field::Field;
use zcash_tachyon::{EpochIndex, keys::private::SpendingKey, pasta::Fp};

use crate::{
    chain::bytes32,
    control::Control,
    payment,
    rpc::Rpc,
    store::{Config, PaymentChannel, Store},
};

/// Summary returned by a PIR sync pass.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PirSyncReport {
    pub first_contact_notes: usize,
    pub follow_up_notes: usize,
    pub channels_opened: usize,
    pub tachygram_confirmed: usize,
    pub tachygram_spent: usize,
    pub earliest_discovered_height: Option<u32>,
}

/// Client for all four PIR products plus the Registry write API.
pub struct PirClient {
    origin: String,
    ca_pem: Option<Vec<u8>>,
    http: HttpClient,
    runtime: Option<tokio::runtime::Runtime>,
    control: Control,
}

impl Drop for PirClient {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

impl PirClient {
    pub fn new(origin: &str, ca_file: Option<&Path>, control: &Control) -> Result<Self> {
        control.check()?;
        let ca_pem = ca_file
            .map(|p| {
                std::fs::read(p).with_context(|| format!("cannot read PIR CA file {}", p.display()))
            })
            .transpose()?;
        let mut builder = HttpClient::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30));
        if let Some(pem) = &ca_pem {
            let certs = reqwest::Certificate::from_pem_bundle(pem)
                .context("invalid PEM certificate in PIR CA file")?;
            ensure!(!certs.is_empty(), "PIR CA file contains no certificates");
            builder = builder.tls_built_in_root_certs(false);
            for cert in certs {
                builder = builder.add_root_certificate(cert);
            }
        }
        let http = builder.build()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            origin: origin.trim_end_matches('/').to_owned(),
            ca_pem,
            http,
            runtime: Some(runtime),
            control: control.clone(),
        })
    }

    fn product_client(&self, product: Product) -> Result<ProductClient> {
        ProductClient::new(&self.origin, product, self.ca_pem.as_deref())
            .map_err(|e| anyhow::anyhow!("PIR client init ({product}): {e}"))
    }

    fn block_on<F, T>(&self, fut: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        self.runtime
            .as_ref()
            .expect("PIR runtime exists until drop")
            .block_on(async {
                tokio::select! {
                    biased;
                    _ = self.control.cancellation.cancelled() => {
                        self.control.check()?;
                        unreachable!("cancellation triggered")
                    }
                    res = fut => res,
                }
            })
    }

    /// Registers `address` (`pk || ek`) with the Registry write API, solving the
    /// proof-of-work challenge locally.
    pub fn register_address(&self, write_origin: &str, address: &Address) -> Result<Registration> {
        let base = format!(
            "{}{}",
            write_origin.trim_end_matches('/'),
            Product::Registry.mount()
        );
        self.control.stage("Fetching registry PoW challenge…")?;
        let challenge_url = format!("{base}/{CHALLENGE_ROUTE}");
        let challenge: Challenge = self.block_on(async {
            let resp = self
                .http
                .get(&challenge_url)
                .send()
                .await
                .context("GET registry challenge failed")?;
            let status = resp.status();
            let bytes = resp.bytes().await?;
            ensure!(
                status.is_success(),
                "registry challenge HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
            );
            Ok(serde_json::from_slice(&bytes)?)
        })?;

        let digest = address.digest();
        self.control.stage(format!(
            "Solving registry PoW ({} bits)…",
            challenge.difficulty_bits
        ))?;
        let nonce = solve_pow(&challenge.challenge.0, &digest, challenge.difficulty_bits)
            .context("could not solve registry proof-of-work")?;
        self.control.check()?;

        let req = RegisterRequest {
            challenge: challenge.challenge.0,
            nonce,
            address: address.clone(),
        };
        let register_url = format!("{base}/{REGISTER_ROUTE}");
        self.control.stage("Submitting registry registration…")?;
        self.block_on(async {
            let resp = self
                .http
                .post(&register_url)
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .body(req.encode())
                .send()
                .await
                .context("POST registry register failed")?;
            let status = resp.status();
            let bytes = resp.bytes().await?;
            ensure!(
                status.is_success(),
                "registry register HTTP {status}: {}",
                String::from_utf8_lossy(&bytes)
            );
            Ok(serde_json::from_slice(&bytes)?)
        })
    }

    /// Privately resolves `(chunk_id, digest)` via `/registry/v1` PIR and verifies
    /// `Address::digest() == digest`.
    pub fn resolve_address(&self, chunk_id: u64, digest: &[u8; 32]) -> Result<Address> {
        self.control.stage(format!(
            "Resolving recipient address via Registry PIR (chunk {chunk_id})…"
        ))?;
        let client = self.product_client(Product::Registry)?;
        let key16: [u8; 16] = digest[..16].try_into().expect("16 bytes");
        let outcome = self.block_on(async {
            let map = client
                .fetch_map()
                .await
                .map_err(|e| anyhow::anyhow!("registry fetch_map failed: {e}"))?;
            client
                .lookup(&map, ScopeSelector::Id(chunk_id), &key16)
                .await
                .map_err(|e| anyhow::anyhow!("registry PIR lookup failed: {e}"))
        })?;
        match outcome {
            LookupOutcome::Found { entries, .. } => {
                for entry in entries {
                    if let Entry::Registry(reg) = entry
                        && reg.address.digest() == *digest
                    {
                        payment::validate_pk_bytes(&reg.address.pk)?;
                        return Ok(reg.address);
                    }
                }
                anyhow::bail!("registry PIR returned no entry matching address digest")
            }
            LookupOutcome::Absent { .. } => {
                anyhow::bail!("recipient address not found in registry chunk {chunk_id}")
            }
            LookupOutcome::BelowFloor { floor_scope, .. } => {
                anyhow::bail!("registry chunk {chunk_id} is below PIR server floor {floor_scope}")
            }
        }
    }

    /// Runs the full private discovery and status pass across `first-contact`,
    /// `memo`, and `tachygram` PIR products.
    pub fn sync_pir(&self, store: &Store, config: &Config, rpc: &Rpc) -> Result<PirSyncReport> {
        let mut report = PirSyncReport::default();
        let own_pk = payment::own_pk_bytes(config.seed);
        let key = SpendingKey::from(config.seed);
        let max_ek_index = config.ek_index.unwrap_or(0);
        let mut cursors = store.pir_cursors()?;

        // 1. First-contact discovery (/first-contact/v1) across all active ek indices 0..=max_ek_index
        self.control.stage("Querying First-Contact PIR…")?;
        let fc_client = self.product_client(Product::FirstContact)?;
        let fc_map = self.block_on(async {
            fc_client
                .fetch_map()
                .await
                .map_err(|e| anyhow::anyhow!("first-contact fetch_map failed: {e}"))
        })?;
        verify_chain_map(&fc_map, config, rpc)?;

        let fc_start = cursors
            .first_contact_sealed_scope
            .map_or(0, |s| s.saturating_add(1))
            .max(fc_map.floor_scope());
        let mut highest_fc_sealed = cursors.first_contact_sealed_scope;
        for ek_index in 0..=max_ek_index {
            let own_addr = payment::own_registry_address(config.seed, ek_index);
            let fc_tag = payment::first_contact_tag(&own_addr.ek);
            for (idx, scope) in fc_map.scopes().iter().enumerate() {
                let scope_idx = idx as u64;
                if scope_idx < fc_start {
                    continue;
                }
                self.control.check()?;
                let outcome = self.block_on(async {
                    fc_client
                        .lookup(&fc_map, ScopeSelector::Index(scope_idx), &fc_tag[..16])
                        .await
                        .map_err(|e| anyhow::anyhow!("first-contact lookup failed: {e}"))
                })?;
                if let LookupOutcome::Found { entries, .. } = outcome {
                    for entry in entries {
                        if let Entry::FirstContact(fc) = entry
                            && let Some((plaintext, channel_key)) =
                                payment::open_first_contact(config.seed, ek_index, &fc)?
                        {
                            let receipt = plaintext.to_receipt(&config.network_id, &own_pk)?;
                            let is_new = store.put_note_if_new(&receipt)?;
                            if is_new {
                                report.first_contact_notes += 1;
                            }
                            let channel_id = channel_id_for("inbound", &channel_key);
                            if store.channel(&channel_id)?.is_none() {
                                store.put_channel(&PaymentChannel {
                                    id: channel_id,
                                    direction: "inbound".into(),
                                    peer_digest: None,
                                    peer_pk: None,
                                    key_hex: hex::encode(channel_key),
                                    next_tag_index: 0,
                                    ek_index: Some(ek_index),
                                    created_scope: Some(scope_idx),
                                })?;
                                report.channels_opened += 1;
                            }
                        }
                    }
                }
                if is_sealed(scope) {
                    highest_fc_sealed =
                        Some(highest_fc_sealed.map_or(scope_idx, |h| h.max(scope_idx)));
                }
            }
        }
        cursors.first_contact_sealed_scope = highest_fc_sealed;

        // 2. Follow-up memo discovery (/memo/v1) for all inbound channels
        let inbound_channels: Vec<PaymentChannel> = store
            .channels()?
            .into_iter()
            .filter(|c| c.direction == "inbound")
            .collect();
        if !inbound_channels.is_empty() {
            self.control
                .stage("Querying Memo PIR for inbound channels…")?;
            let memo_client = self.product_client(Product::Memo)?;
            let memo_map = self.block_on(async {
                memo_client
                    .fetch_map()
                    .await
                    .map_err(|e| anyhow::anyhow!("memo fetch_map failed: {e}"))
            })?;
            verify_chain_map(&memo_map, config, rpc)?;

            let memo_start = cursors
                .memo_sealed_scope
                .map_or(0, |s| s.saturating_add(1))
                .max(memo_map.floor_scope());
            let mut highest_memo_sealed = cursors.memo_sealed_scope;

            for mut channel in inbound_channels {
                let channel_key = bytes32(&channel.key_hex)?;
                let mut next_idx = channel.next_tag_index;
                // Walk scopes from cursor to tip; when a tag hits, keep checking
                // subsequent indices on the same and later scopes until no more hits.
                'advance: loop {
                    let tag = payment::follow_up_tag(&channel_key, next_idx);
                    let mut found_for_idx = false;
                    for (idx, scope) in memo_map.scopes().iter().enumerate() {
                        let scope_idx = idx as u64;
                        if scope_idx < memo_start {
                            continue;
                        }
                        self.control.check()?;
                        let outcome = self.block_on(async {
                            memo_client
                                .lookup(&memo_map, ScopeSelector::Index(scope_idx), &tag[..16])
                                .await
                                .map_err(|e| anyhow::anyhow!("memo lookup failed: {e}"))
                        })?;
                        if let LookupOutcome::Found { entries, .. } = outcome {
                            for entry in entries {
                                if let Entry::Memo(memo) = entry
                                    && let Some(plaintext) = payment::open_follow_up(
                                        &own_pk,
                                        &channel_key,
                                        next_idx,
                                        &memo,
                                    )?
                                {
                                    let receipt =
                                        plaintext.to_receipt(&config.network_id, &own_pk)?;
                                    if store.put_note_if_new(&receipt)? {
                                        report.follow_up_notes += 1;
                                    }
                                    found_for_idx = true;
                                }
                            }
                        }
                        if is_sealed(scope) {
                            highest_memo_sealed =
                                Some(highest_memo_sealed.map_or(scope_idx, |h| h.max(scope_idx)));
                        }
                    }
                    if found_for_idx {
                        next_idx = next_idx.saturating_add(1);
                    } else {
                        break 'advance;
                    }
                }
                if next_idx != channel.next_tag_index {
                    channel.next_tag_index = next_idx;
                    store.put_channel(&channel)?;
                }
            }
            cursors.memo_sealed_scope = highest_memo_sealed;
        }

        // 3. Tachygram inclusion & nullifier exclusion (/tachygram/v1)
        let notes = store.notes()?;
        if !notes.is_empty() {
            self.control
                .stage("Querying Tachygram PIR for note commitments and nullifiers…")?;
            let tg_client = self.product_client(Product::Tachygram)?;
            let tg_map = self.block_on(async {
                tg_client
                    .fetch_map()
                    .await
                    .map_err(|e| anyhow::anyhow!("tachygram fetch_map failed: {e}"))
            })?;
            verify_chain_map(&tg_map, config, rpc)?;

            let floor = tg_map.floor_scope();
            let scopes = tg_map.scopes();
            let mut highest_tg_sealed = cursors.tachygram_sealed_scope;
            for (idx, scope) in scopes.iter().enumerate() {
                if is_sealed(scope) {
                    highest_tg_sealed =
                        Some(highest_tg_sealed.map_or(idx as u64, |h| h.max(idx as u64)));
                }
            }

            for receipt in notes {
                self.control.check()?;
                receipt.check_owner(&key, &config.network_id)?;
                let note_id = receipt.id()?;
                let cm = bytes32(&note_id)?;
                let note = receipt.note()?;
                let master = key.derive_nullifier_private().derive_note_private(note.psi);
                let mut status = store.note_pir_status(&note_id)?.unwrap_or_default();

                // Locate note commitment if not yet confirmed via PIR.
                if status.mined_epoch.is_none() {
                    for (idx, _scope) in scopes.iter().enumerate() {
                        let epoch = idx as u64;
                        if epoch < floor {
                            continue;
                        }
                        let outcome = self.block_on(async {
                            tg_client
                                .lookup(&tg_map, ScopeSelector::Index(epoch), &cm)
                                .await
                                .map_err(|e| anyhow::anyhow!("tachygram cm lookup failed: {e}"))
                        })?;
                        if let LookupOutcome::Found { entries, .. } = outcome
                            && let Some(Entry::Tachygram(tg)) = entries.first()
                        {
                            let epoch_u32 = u32::try_from(epoch)?;
                            status.mined_epoch = Some(epoch_u32);
                            status.mined_height = Some(tg.blk_height);
                            status.stamp_ordinal = Some(tg.stamp_ordinal);
                            report.tachygram_confirmed += 1;
                            report.earliest_discovered_height = Some(
                                report
                                    .earliest_discovered_height
                                    .map_or(tg.blk_height, |h| h.min(tg.blk_height)),
                            );
                            break;
                        }
                    }
                } else if let Some(h) = status.mined_height {
                    report.earliest_discovered_height =
                        Some(report.earliest_discovered_height.map_or(h, |e| e.min(h)));
                }

                // Check nullifiers across epochs from mined_epoch onward if not yet spent.
                if let Some(mined_epoch) = status.mined_epoch
                    && status.spent_epoch.is_none()
                {
                    let start_epoch = status
                        .checked_through_sealed_epoch
                        .map_or(u64::from(mined_epoch), |e| u64::from(e).saturating_add(1))
                        .max(floor);
                    for (idx, scope) in scopes.iter().enumerate() {
                        let epoch = idx as u64;
                        if epoch < start_epoch {
                            continue;
                        }
                        let epoch_u32 = u32::try_from(epoch)?;
                        let nf: [u8; 32] = Fp::from(master.derive_nullifier(EpochIndex::new(epoch_u32)))
                            .to_bytes()
                            .as_slice()
                            .try_into()
                            .expect("32-byte nullifier");
                        let outcome = self.block_on(async {
                            tg_client
                                .lookup(&tg_map, ScopeSelector::Index(epoch), &nf)
                                .await
                                .map_err(|e| anyhow::anyhow!("tachygram nf lookup failed: {e}"))
                        })?;
                        if let LookupOutcome::Found { entries, .. } = outcome
                            && !entries.is_empty()
                        {
                            status.spent_epoch = Some(epoch_u32);
                            report.tachygram_spent += 1;
                            break;
                        }
                        if is_sealed(scope) {
                            status.checked_through_sealed_epoch = Some(epoch_u32);
                        }
                    }
                }

                store.put_note_pir_status(&note_id, &status)?;
            }
            cursors.tachygram_sealed_scope = highest_tg_sealed;
        }

        store.put_pir_cursors(&cursors)?;
        Ok(report)
    }
}

/// Verifies a chain product map's genesis and anchor hashes against the wallet's
/// Zakura node (spec 12.1).
fn verify_chain_map(map: &MapView, config: &Config, rpc: &Rpc) -> Result<()> {
    let map_genesis = map.genesis_hash().display_hex();
    ensure!(
        map_genesis == config.genesis,
        "PIR server genesis {map_genesis} does not match wallet genesis {}",
        config.genesis
    );
    if let MapHead::Chain(head) = map.head() {
        let node_anchor = rpc.hash(head.anchor_height)?;
        let map_anchor = head.anchor_hash.display_hex();
        ensure!(
            node_anchor == map_anchor,
            "PIR server anchor at height {} ({map_anchor}) does not match node ({node_anchor})",
            head.anchor_height
        );
    }
    Ok(())
}

fn is_sealed(scope: &tachyon_pir_cli::ViewScope) -> bool {
    match scope.coverage() {
        tachyon_pir_cli::Coverage::Heights { state, .. }
        | tachyon_pir_cli::Coverage::Sequences { state, .. } => state == ScopeState::Sealed,
    }
}

pub fn channel_id_for(direction: &str, channel_key: &[u8; 32]) -> String {
    let digest = Sha256::new()
        .chain_update(b"tachyon-wallet/channel-id/v1\0")
        .chain_update(direction.as_bytes())
        .chain_update(channel_key)
        .finalize();
    hex::encode(&digest[..16])
}
