# Integer arithmetic
tokens_used = 450
tokens_limit = 1024
remaining = tokens_limit - tokens_used # 574
# Float arithmetic
cost_per_token = 0.000003
total_cost = tokens_used * cost_per_token
print(f"Cost: ${total_cost:.6f}")
# Integer division and modulo
batches = tokens_used // 100 # 4
leftover = tokens_used % 100 # 50
# Built-in math
import math
print(math.log2(512)) # 9.0 - useful in information theory
print(math.ceil(3.1)) # 4