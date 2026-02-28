use crate::error::{Result, VitureError};

pub fn check_status(function: &'static str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(VitureError::SdkCallFailed { function, code })
    }
}

pub fn check_non_negative(function: &'static str, code: i32) -> Result<i32> {
    if code >= 0 {
        Ok(code)
    } else {
        Err(VitureError::SdkCallFailed { function, code })
    }
}
