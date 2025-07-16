import yaml
with open('excavision.yaml') as f:
    data = yaml.safe_load(f)
print(data)
