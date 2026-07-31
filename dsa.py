import random

def coin_toss_simulator():
    while True:
        try:
            n = int(input("How many times do you want to toss the coin? "))
            if n <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    heads = 0
    tails = 0

    for _ in range(n):
        result = random.randint(0, 1)
        if result == 0:
            heads += 1
        else:
            tails += 1

    heads_pct = (heads / n) * 100
    tails_pct = (tails / n) * 100

    print(f"\nResults after {n} tosses:")
    print(f"  Heads: {heads} ({heads_pct:.2f}%)")
    print(f"  Tails: {tails} ({tails_pct:.2f}%)")

coin_toss_simulator()