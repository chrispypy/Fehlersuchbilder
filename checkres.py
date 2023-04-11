with open('res.txt','r') as f:
    l = f.read()
reslist = l.split(',')
print(len(reslist))
reslist.remove('')
reslist = list(set(reslist))
print(f'\n{reslist}\n')
