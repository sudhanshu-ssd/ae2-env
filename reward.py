from grader import _safe_score

def calculate_reward(g_result: dict, difficulty: str) -> float:
    status = g_result["status"]
    tests_passed = g_result["tests_passed"]
    total_tests = g_result["total_tests"]
    eff = g_result["efficiency"]
    
    # Ratios (Handled in Grader: actual / baseline)
    # Note: For speed/mem, lower is better. 
    # So speed_ratio = baseline_ms / runtime_ms (Higher than 1.0 is GOOD)
    speed_ratio = eff.get("speed_ratio", 1.0)
    memory_ratio = eff.get("memory_ratio", 1.0)

    # 1. Hard Fail Penalties (Negative Reinforcement)
    if status == "syntax_error":
        return 0.02
    if status == "runtime_error":
        return 0.02

    # 2. Correctness Score (Linear progression from 0.0 to 0.6)
    # We give more weight to correctness (60% of the total base reward)
    correctness_base = max(0.02, 0.6 * (tests_passed / total_tests))

    # 3. Efficiency Bonus (Only if code is actually correct)
    efficiency_bonus = 0.01
    
    if status == "success":
        # Calculate individual gains (capped at 100% improvement to prevent outliers)
        # speed_score > 0 means it beat the baseline
        speed_gain = min(max(speed_ratio - 1.0, 0.0), 1.0) 
        memory_gain = min(max(memory_ratio - 1.0, 0.0), 1.0)

        # Logic for "Net Improvement"
        if speed_ratio >= 1.0 and memory_ratio >= 1.0:
            # Both improved - Full Bonus (up to 0.4)
            efficiency_bonus = (0.25 * speed_gain) + (0.15 * memory_gain)
        
        elif speed_ratio < 1.0 and memory_ratio < 1.0:
            # Both got worse - Small penalty even if logic is correct
            efficiency_bonus = 0.01
            
        else:
            # Tradeoff scenario (one better, one worse)
            # We allow trading memory for speed, but with a small tax
            net_gain = (0.25 * (speed_ratio - 1.0)) + (0.15 * (memory_ratio - 1.0))
            efficiency_bonus = max(-0.05, net_gain)

    # 4. Total Calculation
    total_base = correctness_base + efficiency_bonus
    
    # 5. Difficulty Scaling (Positive reinforcement for Harder tasks)
    # Instead of multiplying the whole thing (which risks > 1.0), 
    # we use a subtle multiplier that preserves the 0-1 range.
    diff_bonus = {"EASY": 0.02, "MEDIUM": 0.05, "HARD": 0.1}
    final_reward = total_base + diff_bonus.get(difficulty, 0.02)

    # 1. Round first
    rounded_val = round(final_reward, 3)

    # 2. THEN Clamp to ensure we never hit 0.0 or 1.0 even after rounding
    # This prevents 0.0001 from becoming 0.0 and 0.9999 from becoming 1.0
    clamped_val = max(0.02, min(rounded_val, 0.98))
    clamped_val = _safe_score(clamped_val)

    # 3. Final type safety cast
    return float(clamped_val)
