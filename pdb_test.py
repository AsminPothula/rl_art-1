def add(x, y):
    result = x + y
    return result

def calculate(a, b):
    sum_result = add(a, b)
    product_result = a * b
    final_result = sum_result / 0  # Oops, potential division by zero!
    return final_result

if __name__ == "__main__":
    num1 = 5
    num2 = 10
    output = calculate(num1, num2)
    print(f"The final result is: {output}")