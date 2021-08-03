import json

def merge_json(in_files_list, out_file) -> None:
    """
        parameters:
            in_files_list: files to merge
            out_file: file to save merged info
        action:
            create and write merged info to new json
        return:
            None
    """
    tmp = {}
    for path in in_files_list:
        with open(path) as f:
            content = json.load(f)
            for key in content.keys():
                if key not in tmp.keys():
                    tmp[key] = content[key]
                else:
                    tmp[key].extend(content[key])
    with open(out_file,'w') as f:
        outObj = json.dumps(tmp)
        f.write(outObj)

if __name__=='__main__':
    merge_json(['1.json','2.json'],'3.json')