import os
import yaml
root='clean_baseline_dataset'
label_paths=[]
for dirpath,dirs,files in os.walk(root):
    for f in files:
        if f.endswith('.txt'):
            label_paths.append(os.path.join(dirpath,f))
labels=set()
for p in label_paths:
    try:
        with open(p,'r') as fh:
            for line in fh:
                line=line.strip()
                if not line: continue
                parts=line.split()
                try:
                    cls=int(float(parts[0]))
                except:
                    continue
                labels.add(cls)
    except Exception as e:
        print('ERR',p,e)
print('Found',len(label_paths),'label files')
if labels:
    sorted_labels=sorted(labels)
    print('Unique labels (sample 50):', sorted_labels[:50])
    print('Max label index:', max(sorted_labels))
else:
    print('No labels found')
# read configs/base.yaml
cfg='configs/base.yaml'
try:
    c=yaml.safe_load(open(cfg))
    num=c.get('classes',{}).get('num_classes',None)
    print('Config classes.num_classes in',cfg,':',num)
except Exception as e:
    print('Failed reading config',e)
# find any label >= num
if 'num' in locals() and num is not None:
    bad=[]
    for p in label_paths:
        try:
            with open(p) as fh:
                for line in fh:
                    if not line.strip(): continue
                    cls=int(float(line.split()[0]))
                    if cls>=num:
                        bad.append((p,cls))
                        break
        except:
            pass
    print('Files with label >= num_classes:', len(bad))
    if len(bad)>0:
        print('Examples:')
        for p,cls in bad[:10]:
            print(p,cls)
