import redis

r = redis.Redis(host='localhost', port=6379, db=0)
r.set('tes', 'halo dari redis')
print(r.get('tes'))