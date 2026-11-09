import mumo_framework as mf

print("==================================================")
print("   MUMO FRAMEWORK - CUSTOM DATA & EDGE CASE SUITE ")
print("==================================================")

scenarios = {
    "Standard Baseline": [10.0, 12.5, 14.0, 15.5, 18.0, 21.0, 25.0, 28.5, 32.0, 36.0],
    "Extreme Outlier Stress Test": [5.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 1000.0],
    "Mixed / Negative Financial Data": [-15.5, -5.0, 2.0, 8.5, 14.0, 21.0, 19.5]
}

for name, data in scenarios.items():
    print(f"\n--------------------------------------------------")
    print(f"SCENARIO: {name} (n={len(data)})")
    print(f"Data: {data}")
    print(f"--------------------------------------------------")
    
    try:
        # Dynamically set slice size k based on dataset length
        k = 2 if len(data) >= 5 else 1
        alpha = 0.6
        beta = 0.5
        gamma = 0.5

        print(f"-> MPU (Potential Upside): {mf.calculate_mpu(data):.4f}")
        print(f"-> MPD (Potential Downside): {mf.calculate_mpd(data):.4f}")
        print(f"-> MBV (Bounds of Variance, k={k}): {mf.calculate_mbv(data, k, alpha):.4f}")
        print(f"-> P-GMCSI (Core-Squeeze Index): {mf.calculate_p_gmcsi(data, k, alpha):.4f}")
        print(f"-> MAEI (Adaptive Equilibrium): {mf.calculate_maei(data, k, alpha, beta, gamma):.4f}")
        
        params = mf.autotune_parameters(data)
        print(f"-> Auto-tuned Parameters -> k: {params.k}, alpha: {params.alpha:.2f}, beta: {params.beta:.2f}")
        
    except Exception as e:
        print(f"[ERROR] {e}")

print("\n==================================================")
print("        EDGE CASE TESTING COMPLETED SUCCESSFULLY  ")
print("==================================================")