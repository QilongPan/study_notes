# python tutorial

## 文件名模式匹配，过滤文件

### glob

#### 使用案例

```
import glob
for name in glob.glob('dir/file?.txt'):
    print (name)
```

```
dir/file1.txt
dir/file2.txt
dir/filea.txt
dir/fileb.txt
```

## 将二进制字符串'/b'开头转为整数列表

使用案例：

```
def conver2num(bytes_list):
    temp = b''.join(bytes_list)
    data = [x for x in temp]
    return data
```

