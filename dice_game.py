# Bonus Exercise: Dice Strategy
def values(n):
    if n == 1:
        return 3.5
    else:
        vals = [max(x,values(n-1)) for x in range(1,7)]
        return sum(vals)/6
def strategy(n):
    # Compute expected values
    cutoff = [values(k) for k in range(1,n+1)][::-1]
    stops = [[k for k in range(1,7) if k >= cutoff[i]] for i in range(n)]
    return list(zip(cutoff,stops))

# Test cases
print("1 dice:", strategy(1))
print("2 dice:", strategy(2))
print("3 dice:", strategy(3))
print("4 dice:", strategy(4))