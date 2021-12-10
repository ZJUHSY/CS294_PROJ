import torch
import argparse
import random
import os
import json

def main(ratio):
    if ratio == 1.0:
        return 
    
    prefix = './data/train.json'
    write_prefix = prefix + '_' + str(ratio)
    if os.path.isfile(write_prefix + '.dat'):
        return 
    
    data = torch.load(prefix + '.dat')
    sta = torch.load(prefix + '.sta')
    end = torch.load(prefix + '.end')
    lab = torch.load(prefix + '.lab')
    # print(sta) 
    # print(end)
    inp = open('./data/train.json', 'rb')
    passages = json.load(inp)
    all_idxs = list(range(len(lab)))
    sel_idxs = random.sample(all_idxs, int(len(all_idxs) * ratio))

    new_data = None
    new_label = []
    new_start = []
    new_end = []
    raw_data = []
    for sel_idx in sel_idxs:
        # print(sel_idx)
        raw_data.append(passages[sel_idx])
        sta_idx, end_idx = sta[sel_idx], end[sel_idx]
        cur_data = data[sta_idx : end_idx]
        if new_data is None:
            new_start += [0]
            new_data = cur_data
        else:
            new_start += [new_data.shape[0]]
            new_data = torch.cat((new_data, cur_data), dim = 0)
        new_end += [new_data.shape[0]]
        new_label += [lab[sel_idx]]
    inp.close()

    outp = open("./data/train_" + str(ratio) + '.json', 'w', encoding="utf-8")
    outp.write(json.dumps(raw_data, indent=4, ensure_ascii=False))
    outp.close()

    torch.save(new_data, write_prefix + ".dat")
    torch.save(new_label, write_prefix + ".lab")
    torch.save(new_start, write_prefix + ".sta")
    torch.save(new_end, write_prefix + ".end")


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("-rt", "--ratio", type=float, default=1.0)
    args = parser.parse_args()
    main(args.ratio)

  



