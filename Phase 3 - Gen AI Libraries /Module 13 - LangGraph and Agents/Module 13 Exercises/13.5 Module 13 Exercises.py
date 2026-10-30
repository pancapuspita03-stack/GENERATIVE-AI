# 13.5 Module 13 Exercises (Halaman 152)
# Step-limit ReAct check
def check_iteration_limit(step_count: int, max_steps: int = 8) -> bool:
    return step_count >= max_steps

if __name__ == "__main__":
    print("Iteration check (step 9 of 8):", check_iteration_limit(9, 8))
