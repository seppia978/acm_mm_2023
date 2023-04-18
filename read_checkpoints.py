import os
import json
import ast
import itertools

root = '/nas/softechict-nas-2/spoppi/pycharm_projects/inspecting_twin_models/outs/test'
nn = ['vgg16', 'resnet18', 'vit_small_16224', 'vit_tiny_16224']
dataset = ['imagenet', 'cifar10', 'mnist']
baseline = ['standard', 'flipped', 'difference', 'logits']

jsonobj = {k: {f'{n}_{d}':'' for n,d in itertools.product(nn, dataset)} for k in baseline}

null = []
for f in os.listdir(root):
    if f in nn:
        files = [os.path.join(root,f,x) for x in os.listdir(os.path.join(root, f))]
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        files = list(filter(lambda x: os.path.getmtime(x) > 1677974400.0, files)) # filter everything after march 5th 1 am.

        for i, n in enumerate(files[:12]):
            with open(os.path.join(root, f, n), 'r') as file:
                txt = file.readlines()

            params = dict(ast.literal_eval(txt[5]))

            n = f
            d = params['dataset']
            b = 'standard'
            if 'flipped' in params.keys() and params['flipped']:
                b = 'flipped'
            elif params['logits']:
                b = 'logits'
            elif params['loss_type'] == 'difference':
                b = 'difference'
            
            for idx in range(len(txt)):
                if txt[idx].startswith('Saved at'):
                    break
            path = '/'.join(txt[idx][9:].split('/')[:-1])
            if path == '':
                null.append((b, n, d))
            jsonobj[b][f'{n}_{d}'] = path

with open('/mnt/beegfs/work/dnai_explainability/final_ckpts_5-3-23.json', 'w') as f:
    json.dump(jsonobj, f)
print(f'Updated! \nNull {null}')
