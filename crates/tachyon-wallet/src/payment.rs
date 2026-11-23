//! Tachyon payment protocol cryptography and address encodings:
//! deterministic ML-KEM-768 keys, `TPM1` first-contact (`0x02`) and follow-up
//! (`0x01`) memo envelopes, and short/full/legacy address parsing.

use anyhow::{Context, Result, ensure};
use chacha20poly1305::{
    ChaCha20Poly1305, KeyInit, Nonce,
    aead::{Aead, Payload},
};
use ml_kem::{
    B32, Ciphertext, Encoded, EncodedSizeUser, KemCore, MlKem768, MlKem768Params,
    kem::{Decapsulate, DecapsulationKey, Encapsulate, EncapsulationKey},
};
use sha2::{Digest, Sha256};
use tachyon_pir_protocol::{
    ADDR_BYTES, Address, Envelope, EnvelopeEntry, FirstContactEntry, KEM_CT_BYTES, KEM_EK_BYTES,
    MEMO_CT_BYTES, MemoEntry, PK_BYTES,
};
use udon::field::Field;
use zcash_tachyon::{keys::private::SpendingKey, pasta::Fp};

use crate::{chain::bytes32, note::Receipt};

/// Plaintext payload encrypted inside each 128-byte memo ciphertext
/// (`112 + 16-byte Poly1305 tag = 128 = MEMO_CT_BYTES`).
pub const PLAINTEXT_BYTES: usize = 112;
pub const MAX_TEXT_MEMO_BYTES: usize = 37;
const PLAINTEXT_VERSION: u8 = 1;

const KEM_D_DOMAIN: &[u8] = b"tachyon-wallet/ml-kem-768/d/v1\0";
const KEM_Z_DOMAIN: &[u8] = b"tachyon-wallet/ml-kem-768/z/v1\0";
const FC_TAG_DOMAIN: &[u8] = b"tachyon-pir/first-contact/v1/tag\0";
const CHANNEL_KEY_DOMAIN: &[u8] = b"tachyon-pir/channel/v1/key\0";
const FC_AEAD_KEY_DOMAIN: &[u8] = b"tachyon-pir/first-contact/v1/aead-key\0";
const FC_AEAD_NONCE_DOMAIN: &[u8] = b"tachyon-pir/first-contact/v1/aead-nonce\0";
const FC_AAD_DOMAIN: &[u8] = b"tachyon-pir/first-contact/v1/aad\0";
const MEMO_TAG_DOMAIN: &[u8] = b"tachyon-pir/memo/v1/tag\0";
const MEMO_AEAD_KEY_DOMAIN: &[u8] = b"tachyon-pir/memo/v1/aead-key\0";
const MEMO_AEAD_NONCE_DOMAIN: &[u8] = b"tachyon-pir/memo/v1/aead-nonce\0";
const MEMO_AAD_DOMAIN: &[u8] = b"tachyon-pir/memo/v1/aad\0";

/// Supported Tachyon payment address encodings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PaymentAddress {
    /// Compact address resolved privately via `/registry/v1` PIR:
    /// `tachyon-pir:<chunk_id>:<64-hex-digest>`.
    PirShort { chunk_id: u64, digest: [u8; 32] },
    /// Full inline `(pk || ek)` address requiring no registry lookup:
    /// `tachyon-full:<2432-hex-(pk||ek)>`.
    Full(Box<Address>),
    /// Legacy bare payment key requiring out-of-band receipt delivery:
    /// `tachyon-test:<64-hex-pk>`.
    LegacyPk(String),
}

pub fn format_pir_short_address(chunk_id: u64, digest: &[u8; 32]) -> String {
    format!("tachyon-pir:{chunk_id}:{}", hex::encode(digest))
}

pub fn format_full_address(address: &Address) -> String {
    format!("tachyon-full:{}", hex::encode(address.to_bytes()))
}

pub fn parse_payment_address(input: &str) -> Result<PaymentAddress> {
    if let Some(rest) = input.strip_prefix("tachyon-pir:") {
        let (chunk_str, digest_hex) = rest
            .split_once(':')
            .context("expected tachyon-pir:<chunk_id>:<64-hex-digest>")?;
        let chunk_id: u64 = chunk_str
            .parse()
            .context("invalid chunk_id in tachyon-pir address")?;
        let digest = bytes32(digest_hex).context("invalid digest in tachyon-pir address")?;
        return Ok(PaymentAddress::PirShort { chunk_id, digest });
    }
    if let Some(hex_body) = input.strip_prefix("tachyon-full:") {
        let raw = hex::decode(hex_body).context("invalid hex in tachyon-full address")?;
        ensure!(
            raw.len() == ADDR_BYTES,
            "tachyon-full address must be {ADDR_BYTES} bytes, got {}",
            raw.len()
        );
        let addr = Address::from_bytes(&raw)?;
        validate_pk_bytes(&addr.pk)?;
        return Ok(PaymentAddress::Full(Box::new(addr)));
    }
    if input.starts_with("tachyon-test:") {
        let pk = crate::note::decode_address(input)?;
        return Ok(PaymentAddress::LegacyPk(pk));
    }
    anyhow::bail!(
        "expected tachyon-pir:<chunk_id>:<digest>, tachyon-full:<hex>, or tachyon-test:<pk>"
    )
}

pub fn validate_pk_bytes(pk: &[u8; PK_BYTES]) -> Result<Fp> {
    let mut repr = Fp::ZERO.to_bytes();
    repr.as_mut_slice().copy_from_slice(pk);
    Fp::from_bytes(repr).ok_or_else(|| anyhow::anyhow!("non-canonical payment key field encoding"))
}

pub fn own_pk_bytes(seed: [u8; 32]) -> [u8; PK_BYTES] {
    Fp::from(SpendingKey::from(seed).derive_payment_key())
        .to_bytes()
        .as_slice()
        .try_into()
        .expect("32-byte field encoding")
}

/// Deterministically derives the wallet's ML-KEM-768 keypair for `ek_index`.
pub fn derive_kem_keypair(
    seed: [u8; 32],
    ek_index: u32,
) -> (
    DecapsulationKey<MlKem768Params>,
    EncapsulationKey<MlKem768Params>,
) {
    let d: [u8; 32] = Sha256::new()
        .chain_update(KEM_D_DOMAIN)
        .chain_update(seed)
        .chain_update(ek_index.to_le_bytes())
        .finalize()
        .into();
    let z: [u8; 32] = Sha256::new()
        .chain_update(KEM_Z_DOMAIN)
        .chain_update(seed)
        .chain_update(ek_index.to_le_bytes())
        .finalize()
        .into();
    MlKem768::generate_deterministic(&B32::from(d), &B32::from(z))
}

/// Returns the wallet's `(pk, ek)` registry address for `ek_index`.
pub fn own_registry_address(seed: [u8; 32], ek_index: u32) -> Address {
    let pk = own_pk_bytes(seed);
    let (_dk, ek) = derive_kem_keypair(seed, ek_index);
    let ek_bytes: [u8; KEM_EK_BYTES] = ek.as_bytes().as_slice().try_into().expect("ek 1184 bytes");
    Address { pk, ek: ek_bytes }
}

/// 32-byte first-contact tag `H(ek)` whose first 16 bytes are the PIR key.
pub fn first_contact_tag(ek: &[u8; KEM_EK_BYTES]) -> [u8; 32] {
    Sha256::new()
        .chain_update(FC_TAG_DOMAIN)
        .chain_update(ek)
        .finalize()
        .into()
}

/// 32-byte follow-up memo tag `H(K, index)` whose first 16 bytes are the PIR key.
pub fn follow_up_tag(channel_key: &[u8; 32], index: u64) -> [u8; 32] {
    Sha256::new()
        .chain_update(MEMO_TAG_DOMAIN)
        .chain_update(channel_key)
        .chain_update(index.to_le_bytes())
        .finalize()
        .into()
}

fn derive_channel_key(
    shared_secret: &[u8; 32],
    pk: &[u8; PK_BYTES],
    ek: &[u8; KEM_EK_BYTES],
) -> [u8; 32] {
    Sha256::new()
        .chain_update(CHANNEL_KEY_DOMAIN)
        .chain_update(shared_secret)
        .chain_update(pk)
        .chain_update(ek)
        .finalize()
        .into()
}

fn fc_aead_key_nonce(channel_key: &[u8; 32]) -> ([u8; 32], [u8; 12]) {
    let key: [u8; 32] = Sha256::new()
        .chain_update(FC_AEAD_KEY_DOMAIN)
        .chain_update(channel_key)
        .finalize()
        .into();
    let nonce_full: [u8; 32] = Sha256::new()
        .chain_update(FC_AEAD_NONCE_DOMAIN)
        .chain_update(channel_key)
        .finalize()
        .into();
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&nonce_full[..12]);
    (key, nonce)
}

fn memo_aead_key_nonce(channel_key: &[u8; 32], index: u64) -> ([u8; 32], [u8; 12]) {
    let key: [u8; 32] = Sha256::new()
        .chain_update(MEMO_AEAD_KEY_DOMAIN)
        .chain_update(channel_key)
        .chain_update(index.to_le_bytes())
        .finalize()
        .into();
    let nonce_full: [u8; 32] = Sha256::new()
        .chain_update(MEMO_AEAD_NONCE_DOMAIN)
        .chain_update(channel_key)
        .chain_update(index.to_le_bytes())
        .finalize()
        .into();
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&nonce_full[..12]);
    (key, nonce)
}

fn build_aad(domain: &[u8], tag: &[u8; 32], pk: &[u8; PK_BYTES]) -> Vec<u8> {
    let mut aad = Vec::with_capacity(domain.len() + 32 + PK_BYTES);
    aad.extend_from_slice(domain);
    aad.extend_from_slice(tag);
    aad.extend_from_slice(pk);
    aad
}

/// Fixed-size 112-byte note plaintext carried inside a 128-byte AEAD ciphertext.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotePlaintext {
    pub value_zat: u64,
    pub psi: [u8; 32],
    pub rcm: [u8; 32],
    pub text_memo: Option<String>,
}

impl NotePlaintext {
    pub fn from_receipt(receipt: &Receipt, text_memo: Option<&str>) -> Result<Self> {
        let memo = match text_memo {
            Some(text) => {
                ensure!(
                    text.len() <= MAX_TEXT_MEMO_BYTES,
                    "memo text exceeds {MAX_TEXT_MEMO_BYTES} UTF-8 bytes"
                );
                Some(text.to_owned())
            }
            None => None,
        };
        Ok(Self {
            value_zat: receipt.value_zat,
            psi: bytes32(&receipt.psi)?,
            rcm: bytes32(&receipt.rcm)?,
            text_memo: memo,
        })
    }

    pub fn to_receipt(&self, network_id: &str, own_pk: &[u8; PK_BYTES]) -> Result<Receipt> {
        let receipt = Receipt {
            version: 1,
            network_id: network_id.to_owned(),
            payment_key: hex::encode(own_pk),
            value_zat: self.value_zat,
            psi: hex::encode(self.psi),
            rcm: hex::encode(self.rcm),
            locator: None,
        };
        receipt.note()?;
        Ok(receipt)
    }

    pub fn encode(&self) -> Result<[u8; PLAINTEXT_BYTES]> {
        let memo_bytes = self.text_memo.as_deref().map(str::as_bytes).unwrap_or(&[]);
        ensure!(
            memo_bytes.len() <= MAX_TEXT_MEMO_BYTES,
            "memo text exceeds {MAX_TEXT_MEMO_BYTES} bytes"
        );
        let mut out = [0u8; PLAINTEXT_BYTES];
        out[0] = PLAINTEXT_VERSION;
        out[1] = 0; // flags
        out[2..10].copy_from_slice(&self.value_zat.to_le_bytes());
        out[10..42].copy_from_slice(&self.psi);
        out[42..74].copy_from_slice(&self.rcm);
        out[74] = u8::try_from(memo_bytes.len()).expect("checked <= 37");
        out[75..75 + memo_bytes.len()].copy_from_slice(memo_bytes);
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() == PLAINTEXT_BYTES,
            "invalid note plaintext length"
        );
        ensure!(
            bytes[0] == PLAINTEXT_VERSION,
            "unsupported note plaintext version {}",
            bytes[0]
        );
        let value_zat = u64::from_le_bytes(bytes[2..10].try_into().expect("8 bytes"));
        let psi: [u8; 32] = bytes[10..42].try_into().expect("32 bytes");
        let rcm: [u8; 32] = bytes[42..74].try_into().expect("32 bytes");
        let memo_len = usize::from(bytes[74]);
        ensure!(memo_len <= MAX_TEXT_MEMO_BYTES, "invalid memo length");
        let text_memo = if memo_len == 0 {
            None
        } else {
            Some(
                std::str::from_utf8(&bytes[75..75 + memo_len])
                    .context("memo text is not valid UTF-8")?
                    .to_owned(),
            )
        };
        Ok(Self {
            value_zat,
            psi,
            rcm,
            text_memo,
        })
    }
}

/// Builds a `0x02` first-contact envelope entry and returns `(entry, channel_key)`.
pub fn build_first_contact(
    recipient: &Address,
    note: &NotePlaintext,
) -> Result<(EnvelopeEntry, [u8; 32])> {
    let encoded_ek: Encoded<EncapsulationKey<MlKem768Params>> = recipient.ek.into();
    let ek = EncapsulationKey::<MlKem768Params>::from_bytes(&encoded_ek);
    let mut rng = rand_08::rngs::OsRng;
    let (kem_ct, shared_secret) = ek
        .encapsulate(&mut rng)
        .map_err(|_| anyhow::anyhow!("ML-KEM-768 encapsulation failed"))?;
    let kem_ct_bytes: [u8; KEM_CT_BYTES] = kem_ct
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("unexpected ML-KEM ciphertext size"))?;
    let ss_bytes: [u8; 32] = shared_secret
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("unexpected ML-KEM shared secret size"))?;

    let tag = first_contact_tag(&recipient.ek);
    let channel_key = derive_channel_key(&ss_bytes, &recipient.pk, &recipient.ek);
    let (aead_key, nonce) = fc_aead_key_nonce(&channel_key);
    let aad = build_aad(FC_AAD_DOMAIN, &tag, &recipient.pk);
    let plaintext = note.encode()?;
    let ciphertext_vec = ChaCha20Poly1305::new(&aead_key.into())
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &plaintext,
                aad: &aad,
            },
        )
        .map_err(|_| anyhow::anyhow!("first-contact AEAD encryption failed"))?;
    let ciphertext: [u8; MEMO_CT_BYTES] = ciphertext_vec
        .try_into()
        .map_err(|_| anyhow::anyhow!("unexpected first-contact ciphertext length"))?;

    Ok((
        EnvelopeEntry::FirstContact {
            tag,
            kem_ciphertext: Box::new(kem_ct_bytes),
            ciphertext: Box::new(ciphertext),
        },
        channel_key,
    ))
}

/// Attempts to open a `FirstContactEntry` retrieved from `/first-contact/v1` PIR.
/// Returns `Ok(None)` when the entry does not authenticate under `(seed, ek_index)`.
pub fn open_first_contact(
    seed: [u8; 32],
    ek_index: u32,
    entry: &FirstContactEntry,
) -> Result<Option<(NotePlaintext, [u8; 32])>> {
    let own_addr = own_registry_address(seed, ek_index);
    let tag = first_contact_tag(&own_addr.ek);
    if entry.key != tag[..16] {
        return Ok(None);
    }
    let (dk, _ek) = derive_kem_keypair(seed, ek_index);
    let kem_ct = Ciphertext::<MlKem768>::from(entry.kem_ciphertext);
    let shared_secret = dk
        .decapsulate(&kem_ct)
        .map_err(|_| anyhow::anyhow!("ML-KEM-768 decapsulation failed"))?;
    let ss_bytes: [u8; 32] = shared_secret
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("unexpected ML-KEM shared secret size"))?;
    let channel_key = derive_channel_key(&ss_bytes, &own_addr.pk, &own_addr.ek);
    let (aead_key, nonce) = fc_aead_key_nonce(&channel_key);
    let aad = build_aad(FC_AAD_DOMAIN, &tag, &own_addr.pk);
    let Ok(plaintext) = ChaCha20Poly1305::new(&aead_key.into()).decrypt(
        Nonce::from_slice(&nonce),
        Payload {
            msg: &entry.ciphertext,
            aad: &aad,
        },
    ) else {
        return Ok(None);
    };
    let note = NotePlaintext::decode(&plaintext)?;
    Ok(Some((note, channel_key)))
}

/// Builds a `0x01` follow-up memo envelope entry for an established channel.
pub fn build_follow_up(
    recipient_pk: &[u8; PK_BYTES],
    channel_key: &[u8; 32],
    index: u64,
    note: &NotePlaintext,
) -> Result<EnvelopeEntry> {
    let tag = follow_up_tag(channel_key, index);
    let (aead_key, nonce) = memo_aead_key_nonce(channel_key, index);
    let aad = build_aad(MEMO_AAD_DOMAIN, &tag, recipient_pk);
    let plaintext = note.encode()?;
    let ciphertext_vec = ChaCha20Poly1305::new(&aead_key.into())
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &plaintext,
                aad: &aad,
            },
        )
        .map_err(|_| anyhow::anyhow!("follow-up AEAD encryption failed"))?;
    let ciphertext: [u8; MEMO_CT_BYTES] = ciphertext_vec
        .try_into()
        .map_err(|_| anyhow::anyhow!("unexpected follow-up ciphertext length"))?;

    Ok(EnvelopeEntry::FollowUp {
        tag,
        ciphertext: Box::new(ciphertext),
    })
}

/// Attempts to open a `MemoEntry` retrieved from `/memo/v1` PIR for `(channel_key, index)`.
pub fn open_follow_up(
    own_pk: &[u8; PK_BYTES],
    channel_key: &[u8; 32],
    index: u64,
    entry: &MemoEntry,
) -> Result<Option<NotePlaintext>> {
    let tag = follow_up_tag(channel_key, index);
    if entry.key != tag[..16] {
        return Ok(None);
    }
    let (aead_key, nonce) = memo_aead_key_nonce(channel_key, index);
    let aad = build_aad(MEMO_AAD_DOMAIN, &tag, own_pk);
    let Ok(plaintext) = ChaCha20Poly1305::new(&aead_key.into()).decrypt(
        Nonce::from_slice(&nonce),
        Payload {
            msg: &entry.ciphertext,
            aad: &aad,
        },
    ) else {
        return Ok(None);
    };
    Ok(Some(NotePlaintext::decode(&plaintext)?))
}

/// Encodes one or more entries into a canonical `TPM1` `vMemoTachyon` payload.
pub fn encode_envelope(entries: Vec<EnvelopeEntry>) -> Result<Vec<u8>> {
    Ok(Envelope { entries }.encode()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_contact_and_follow_up_round_trip() {
        let bob_seed = [0x42u8; 32];
        let bob_addr = own_registry_address(bob_seed, 0);
        let note1 = NotePlaintext {
            value_zat: 123_456,
            psi: [0x11; 32],
            rcm: [0x22; 32],
            text_memo: Some("hello tachyon".into()),
        };

        let (fc_entry, alice_k) = build_first_contact(&bob_addr, &note1).unwrap();
        let bytes = encode_envelope(vec![fc_entry.clone()]).unwrap();
        let parsed = tachyon_pir_protocol::envelope::parse(&bytes).unwrap();
        assert_eq!(parsed.entries.len(), 1);
        let table_fc = parsed.entries[0].first_contact_entry().unwrap();

        let (opened1, bob_k) = open_first_contact(bob_seed, 0, &table_fc)
            .unwrap()
            .expect("bob opens first contact");
        assert_eq!(opened1, note1);
        assert_eq!(bob_k, alice_k);

        // Wrong seed cannot decrypt.
        assert!(
            open_first_contact([0x99u8; 32], 0, &table_fc)
                .unwrap()
                .is_none()
        );

        // Follow-up memo on the established channel.
        let note2 = NotePlaintext {
            value_zat: 789_000,
            psi: [0x33; 32],
            rcm: [0x44; 32],
            text_memo: None,
        };
        let fu_entry = build_follow_up(&bob_addr.pk, &alice_k, 0, &note2).unwrap();
        let fu_bytes = encode_envelope(vec![fu_entry]).unwrap();
        let parsed_fu = tachyon_pir_protocol::envelope::parse(&fu_bytes).unwrap();
        let table_memo = parsed_fu.entries[0].memo_entry().unwrap();

        let opened2 = open_follow_up(&bob_addr.pk, &bob_k, 0, &table_memo)
            .unwrap()
            .expect("bob opens follow-up");
        assert_eq!(opened2, note2);
        // Wrong index fails.
        assert!(
            open_follow_up(&bob_addr.pk, &bob_k, 1, &table_memo)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn address_round_trips() {
        let addr = own_registry_address([7u8; 32], 0);
        let digest = addr.digest();
        let short = format_pir_short_address(3, &digest);
        assert_eq!(
            parse_payment_address(&short).unwrap(),
            PaymentAddress::PirShort {
                chunk_id: 3,
                digest
            }
        );
        let full = format_full_address(&addr);
        assert_eq!(
            parse_payment_address(&full).unwrap(),
            PaymentAddress::Full(Box::new(addr))
        );
    }
}
