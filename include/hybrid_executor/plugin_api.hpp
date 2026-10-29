#pragma once

// Public conventions shared by example OpPackages. QNN-specific tensor
// validation remains in each adapter because supported tensor APIs vary by
// QAIRT release.

#define HYEXEC_QNN_SUCCESS_STATUS GraphStatus::Success
