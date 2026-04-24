tokens = ["a", "b", "c", "d", "e", "f"]
chunk_size = 3
overlap = 2

res = []
start = 0
while start + chunk_size <= len(tokens):
    res.append(tokens[start: start + chunk_size])
    start += chunk_size - overlap
print(res)