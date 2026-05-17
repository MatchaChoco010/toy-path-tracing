pub mod management;
mod transfer;

pub use transfer::{
    linear_to_srgb, linear_to_srgb_channel, srgb_to_linear, srgb_to_linear_channel,
};
