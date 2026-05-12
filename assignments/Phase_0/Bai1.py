a = ["the quick brown fox", "the lazy dog sleeps", "the fox jumps over the dog"]
a = " ".join(a).split()
res = {}
for i in a:
    res[i] = res.get(i, 0) + 1
print(res)

# from collections import Counter
# a = ["the quick brown fox", "the lazy dog sleeps", "the fox jumps over the dog"]
# res = dict(Counter(" ".join(a).split()))
# print(res)