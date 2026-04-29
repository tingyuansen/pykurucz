#!/usr/bin/env python3
"""
Trace TK computation to understand what it represents in fort.10.

Key finding: DO 1516 loop that computes TK = k_B*T is NOT called
when reading DECK6 format. So TK must be computed elsewhere or
is uninitialized.
"""

from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from synthe_py.tools.convert_atm_to_npz import _read_fort10_record, parse_atm_file
import numpy as np

def trace_tk_computation():
    """Trace how TK is computed and what it represents."""
    
    print("=" * 80)
    print("TRACING TK COMPUTATION")
    print("=" * 80)
    
    # Read fort.10
    fort10_path = Path('synthe/stmp_at12_aaaaa/fort.10')
    with fort10_path.open('rb') as fh:
        records = []
        while True:
            try:
                records.append(_read_fort10_record(fh))
            except EOFError:
                break
    
    state_block = records[3]
    n_layers = 80
    idx = 0
    T = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    TKEV = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    TK = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    HKT = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    TLOG = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    HCKT = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    P = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    XNE = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    XNATOM = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    RHO = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    RHOX = np.frombuffer(state_block, dtype='<f8', count=n_layers, offset=idx*8); idx += n_layers
    
    print("\n=== Layer 0 Values from fort.10 ===")
    print(f"T:        {T[0]:.6e}")
    print(f"TKEV:     {TKEV[0]:.6e}")
    print(f"TK:       {TK[0]:.6e}")
    print(f"HKT:      {HKT[0]:.6e}")
    print(f"TLOG:     {TLOG[0]:.6e}")
    print(f"HCKT:     {HCKT[0]:.6e}")
    print(f"P:        {P[0]:.6e}")
    print(f"XNE:      {XNE[0]:.6e}")
    print(f"XNATOM:   {XNATOM[0]:.6e}")
    print(f"RHOX:     {RHOX[0]:.6e}")
    
    # Compute expected values
    K_BOLTZ_FORTRAN = 1.38054e-16
    TK_kB_T = K_BOLTZ_FORTRAN * T[0]
    TKEV_expected = T[0] / 11604.518
    HKT_expected = 6.6256e-27 / TK_kB_T
    TLOG_expected = np.log(T[0])
    
    print("\n=== Expected Values (from T) ===")
    print(f"TK = k_B*T:        {TK_kB_T:.6e}")
    print(f"TKEV = T/11604.5:  {TKEV_expected:.6e}")
    print(f"HKT = H_PLANCK/TK: {HKT_expected:.6e}")
    print(f"TLOG = log(T):     {TLOG_expected:.6e}")
    
    print("\n=== Comparison ===")
    print(f"TK:     fort.10={TK[0]:.6e}, expected={TK_kB_T:.6e}, ratio={TK[0]/TK_kB_T:.6e}")
    print(f"TKEV:   fort.10={TKEV[0]:.6e}, expected={TKEV_expected:.6e}")
    print(f"HKT:    fort.10={HKT[0]:.6e}, expected={HKT_expected:.6e}, ratio={HKT[0]/HKT_expected:.6e}")
    print(f"TLOG:   fort.10={TLOG[0]:.6e}, expected={TLOG_expected:.6e}")
    
    # Check if TK can be computed from HKT
    H_PLANCK_ATLAS = 6.6256e-27
    TK_from_HKT = H_PLANCK_ATLAS / HKT[0] if HKT[0] > 0 else 0
    print(f"\n=== TK from HKT ===")
    print(f"TK = H_PLANCK / HKT = {TK_from_HKT:.6e}")
    print(f"TK from fort.10: {TK[0]:.6e}")
    print(f"Match: {abs(TK_from_HKT - TK[0]) < 1e-6}")
    
    # Check XNATOM formula
    P_from_xnatom = (XNATOM[0] + XNE[0]) * TK[0]
    print(f"\n=== XNATOM Formula Check ===")
    print(f"XNATOM = P/TK - XNE")
    print(f"So: P = (XNATOM + XNE) * TK")
    print(f"P from formula: {P_from_xnatom:.6e}")
    print(f"P from fort.10:  {P[0]:.6e}")
    print(f"Match: {abs(P_from_xnatom - P[0]) < 1e-6}")
    
    # Check if TK = P / (XNATOM + XNE) when P is non-zero
    if (XNATOM[0] + XNE[0]) > 0 and P[0] > 0:
        TK_from_p = P[0] / (XNATOM[0] + XNE[0])
        print(f"\n=== TK from P ===")
        print(f"TK = P / (XNATOM + XNE) = {TK_from_p:.6e}")
        print(f"TK from fort.10: {TK[0]:.6e}")
        print(f"Match: {abs(TK_from_p - TK[0]) < 1e-6}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    print("1. TK in fort.10 (0.702) does NOT match k_B*T (5.1e-13)")
    print("2. TKEV, TLOG in fort.10 are zero (should be non-zero)")
    print("3. HKT in fort.10 gives TK = 9.9e-15 (close to k_B*T!)")
    print("4. This suggests HKT might be correct, but TK is wrong")
    print("5. OR: TK represents something completely different")
    print("\nHYPOTHESIS:")
    print("  - TK in fort.10 might be in different units")
    print("  - OR: TK is computed differently (not k_B*T)")
    print("  - OR: TK is uninitialized/garbage from COMMON block")

if __name__ == '__main__':
    trace_tk_computation()

