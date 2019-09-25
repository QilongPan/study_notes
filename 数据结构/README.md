# introduction-to-algorithm
# 算法在计算中的作用

非形式地说，**算法**就是任何良定义的计算过程，该过程取某个值或值得集合作为输入并产生某个值或值的集合作为输出。

# 算法基础

## 插入排序

```
def insertion_sort(arr,descent):
    for j in range(1,len(arr)):
        key = arr[j]
        i = j -1
        if descent:
            while i >= 0 and arr[i] < key:
                arr[i+1] = arr[i]
                i = i - 1
        else:
            while i >= 0 and arr[i] > key:
                arr[i+1] = arr[i]
                i = i - 1
        arr[i+1] = key
```

插入排序最坏情况运行时间为$$\Theta(n^{2})$$。插入排序是稳定的。

## 归并排序

```
def merge_sort(arr,left,right,descent):
    if left < right:
        middle = (right + left) // 2
        merge_sort(arr,left,middle,descent)
        merge_sort(arr,middle+1,right,descent)
        merge(arr,left,middle,right,descent)

def merge(arr,left,middle,right,descent):
    left_arr_len = middle - left + 1
    right_arr_len = right - middle
    left_arr = []
    right_arr = []
    for i in range(left_arr_len):
        left_arr.append(arr[left+i])
    for i in range(right_arr_len):
        right_arr.append(arr[middle+1+i])
    i = 0
    j = 0
    k = left
    while i < left_arr_len and j < right_arr_len:
        if descent:
            if left_arr[i] >= right_arr[j]:
                arr[k] = left_arr[i]
                i = i + 1
            else:
                arr[k] = right_arr[j]
                j = j + 1               
        else:
            if left_arr[i] <= right_arr[j]:
                arr[k] = left_arr[i]
                i = i + 1
            else:
                arr[k] = right_arr[j]
                j = j + 1        
        k = k + 1
    while i < left_arr_len:
        arr[k] = left_arr[i]
        i += 1
        k += 1
    while j < right_arr_len:
        arr[k] = right_arr[j]
        j += 1
        k += 1    
```

时间复杂度为$$\Theta(nlgn)$$。归并排序是稳定的。

# 函数的增长

## 渐进记号

### $\theta$记号

$$\theta$$记号给出一个函数的上界和下界。

对一个给定的函数$$g(n)$$，用$$\theta(g(n))$$来表示以下函数的集合：

$$\theta(g(n))=$${$$f(n)：存在正常量c_{1}、c_{2}和n_{0},使得对所有n\geq n_{0},有0\leq c_{1}g(n)\leq f(n)\leq c_{2}g(n)$$}

### $O$记号

当只有一个渐近上界时，使用$$O$$记号。

对一个给定的函数$$g(n)$$，用$$Og(n)$$来表示以下函数的集合：

$$O(g(n))=$${$$f(n):存在正常量c和n_{0}，使得对所有n\geq n_{0},有0\leq f(n)\leq cg(n)$$}。

### $\Omega $记号

当只有一个渐近下界时，使用$$\Omega$$记号。

$$\Omega(g(n))=$${$$f(n):存在正常量c和n_{0}，使得对所有n\geq n_{0},有0\leq  cg(n)\leq f(n)$$}。

### $o$记号

由$$O$$记号提供的渐近上界可能是也可能不是渐近紧确的。界$$2n^2=O(n^2)$$是渐近紧确的，但是界$$2n=O(n^2)$$却不是。形式化地定义$$o(g(n))$$为以下集合：

$$o(g(n))=$${$$f(n):对任意正常量c> 0，存在常量n_{0}>0，使得对所有n\geq n_{0},有0\leq f(n)\leq cg(n)$$}。例如，$$2n=o(n^2)$$,但是$$2n^2\neq o(n^2)$$.

### $w$记号

$$w$$记号与$$\Omega $$记号的关系类似于$$o$$记号与$$O$$记号的关系。

$$w(g(n))=$${$$f(n):对任意正常量c> 0，存在常量n_{0}>0，使得对所有n\geq n_{0},有0\leq cg(n)\leq f(n) $$}。例如，$$n^2/2=w(n)$$,但是$$n^2/2\neq w(n^2)$$。

# 分治策略

在分治策略中，我们递归地求解一个问题，在每层递归中应用如下三个步骤：

- **分解**步骤将问题划分为一些子问题，子问题的形式与原问题一样，只是规模更小。
- **解决**步骤递归地求解出子问题。如果子问题的规模足够小，则停止递归，直接求解。
- **合并**步骤将子问题的解组合成原问题的解。

## 最大子数组问题

案例:

```
13 -3 -25 20 -3 -16 -23 18 20 -7 12 -5 -22 15 -4 7
```

最大子数组为18 20 -7 12

### 使用分治策略的求解办法

假定我们要寻找子数组$$A[low..high]$$的最大子数组。使用分治技术意味着我们要将子数组划分为两个规模尽量相等的子数组。也就是说，找到子数组的中央位置，比如mid，然后考虑求解两个子数组$$A[low..mid]$$和$$A[mid+1..high]$$。$$A[low..high]$$的任何连续子数组$$A[i..j]$$所处的位置必然是以下三种情况之一：

- 完全位于子数组$$A[low..mid]$$中
- 完全位于子数组$$A[mid+1..high]$$中
- 跨越了中点

```
def maximum_subarray(arr,low,high):
    if low == high:
        return low,high,arr[low]
    else:
        mid = (low+high)//2
        left_low,left_high,left_sum = maximum_subarray(arr,low,mid)
        right_low,right_high,right_sum = maximum_subarray(arr,mid+1,high)
        cross_low,cross_high,cross_sum = find_max_crossing_subarray(arr,low,mid,high)
        if left_sum >= right_sum and left_sum >= cross_sum:
            return left_low,left_high,left_sum
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return right_low,right_high,right_sum
        else:
            return cross_low,cross_high,cross_sum

def find_max_crossing_subarray(arr,low,mid,high):
    arr_sum = 0
    left_sum = None
    max_left = None
    for i in range(mid,low-1,-1):
        arr_sum += arr[i]
        if left_sum == None:
            left_sum = arr_sum
            max_left = i
        else:
            if arr_sum > left_sum:
                left_sum = arr_sum
                max_left = i 
    right_sum = None
    arr_sum = 0
    max_right = None
    for i in range(mid+1,high+1):
        arr_sum += arr[i]
        if right_sum == None:
            right_sum = arr_sum
            max_right = i
        else:
            if arr_sum > right_sum:
                right_sum = arr_sum
                max_right = i
    return max_left,max_right,left_sum+right_sum
```

时间复杂度为$$\Theta(nlgn)$$。

## 矩阵乘法的$Strassen$算法

为简单起见，当使用分治算法计算矩阵积$$C=A\cdot B$$时，假定三个矩阵均为$$n\times n$$矩阵，其中n为2的幂。当n不为2的幂时，需要进行变形。
$$
A = \begin{bmatrix}
A_{11} & A_{12}\\ 
A_{21} & A_{22}
\end{bmatrix},B=\begin{bmatrix}
B_{11} &B_{12} \\ 
B_{21} & B_{22}
\end{bmatrix},C = \begin{bmatrix}
C_{11} &C_{12} \\ 
C_{21} & C_{22}
\end{bmatrix} \\
$$
因此可以将公式$$C=A\cdot B$$改写为：
$$
\begin{bmatrix}
C_{11} &C_{12} \\ 
C_{21} & C_{22}
\end{bmatrix}=\begin{bmatrix}
A_{11} & A_{12}\\ 
A_{21} & A_{22}
\end{bmatrix}\cdot \begin{bmatrix}
B_{11} &B_{12} \\ 
B_{21} & B_{22}
\end{bmatrix}\\
$$
等价于如下4个公式：
$$
\begin{aligned}
C_{11}=A_{11}\cdot B_{11}+A_{12}\cdot B_{21}\\
C_{12}=A_{11}\cdot B_{12}+A_{12}\cdot B_{22}\\
C_{21}=A_{21}\cdot B_{11}+A_{22}\cdot B_{21}\\
C_{22}=A_{21}\cdot B_{12}+A_{22}\cdot B_{22}
\end{aligned}
$$
$$strassen$$算法之所以要n为2的幂的原因在于矩阵二分后，两个子矩阵加减法要满足相同规模。

当n不为2的幂时:把矩阵补全成为2的幂次规模即可。由于矩阵乘法性质，就算扩大矩阵（补0），也会保留原有的结果，而其他部分为0，也就是说算完之后再从结果矩阵将需要部分扣下来即可。


# 概率分析和随机算法

当分析一个随机算法的运行时间时，我们以运行时间的期望值衡量，其中输入值由随机数生成器产生。我们将一个随机算法的运行时间称为期望运行时间。

## 雇用问题

**概率分析：**我们对所有可能输入产生的运行时间取平均。当报告此种类型的运行时间时，我们称其为平均情况运行时间。

## 指示器随机变量

# 堆排序

堆是一个数组，它可以被看成一个近似的完全二叉树。树上的每一个结点对应数组中的一个元素。除了最底层外，该树是完全充满的，而且是从左向右填充。

在堆中很容易计算得到它父节点、左孩子和右孩子的下标：

$$PARENT(i)$$
     $$return \left \lfloor i/2 \right \rfloor$$



$$LEFT(i)$$

​    $$return 2i$$



$$RIGHT(i)$$

​    $$return 2i+1$$



在最大堆中，最大堆性质是指除了根以外的所有结点$$i$$都要满足：
$$
A[PARENT(i)]\geq A[i]
$$
最小堆性质是除了根以外的所有结点$$i$$都有
$$
A[PARENT(i)]\leq  A[i]
$$

如果把堆看成是一棵树，我们定义一个堆中结点的**高度**就为该结点到叶子结点最长简单路径上边的树木；进而我们可以把堆的高度定义为根结点的高度。既然一个包含$$n$$个元素的堆可以看做一棵完全二叉树，那么该堆的高度是$$\theta lg(n)$$。我们会发现，堆结构上的一些基本操作的运行时间至多与树的高度成正比，即时间复杂度为$$O(lgn)$$。

## 维护堆的性质

MAX-HEAPIFY是用于维护最大堆性质的重要过程，其时间复杂度为$$O(lgn)$$。

```
    #在堆排序时，会把最大值与堆的最后一个结点交换，即将根结点移除，但其实根结点还在堆数组中，
    #所以得设置堆大小，避免重建堆时，又将其加入到堆中
    def max_adus_heapify(self,arr,i,heap_size):
        left = self.left(i)
        right = self.right(i)
        if left < heap_size and arr[left]>arr[i]:
            largest = left
        else:
            largest = i 
        if right < heap_size and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            temp = arr[i]
            arr[i] = arr[largest]
            arr[largest] = temp
            self.max_adus_heapify(arr,largest,heap_size)
```

## 建堆

```
    '''
    至底向上构建最大堆，因为i>arr.length/2下标出的结点都为叶结点，所以可以从arr.length/2开始
    '''
    def build_max_heapify(self,arr):
        for i in range(len(arr)-1,-1,-1):
            self.max_adus_heapify(arr,i,len(arr))

    '''
```

## 堆排序

```
    '''
    首先构建最大堆，使用最大堆进行堆排序
    将最大堆中根结点（最大值）与堆末尾结点交换,因为交换后的堆可能不是最大堆了，所以得重建堆，但是堆的大小将减少1
    '''
    def heap_sort(self,arr):
        heap_size = len(arr)
        #构建最大堆
        self.build_max_heapify(arr)
        for i in range(len(arr)-1,0,-1):
            temp = arr[i]
            arr[i] = arr[0]
            arr[0] = temp
            heap_size -= 1
            self.max_adus_heapify(arr,0,heap_size)
```



```
class MaxHeapify:

    def __init__(self):
        pass
        
    def parent(self,i):
        return (i-1)//2

    def left(self,i):
        return 2*(i+1)-1

    def right(self,i):
        return 2*(i+1)

    #在堆排序时，会把最大值与堆的最后一个结点交换，即将根结点移除，但其实根结点还在堆数组中，
    #所以得设置堆大小，避免重建堆时，又将其加入到堆中
    def max_adus_heapify(self,arr,i,heap_size):
        left = self.left(i)
        right = self.right(i)
        if left < heap_size and arr[left]>arr[i]:
            largest = left
        else:
            largest = i 
        if right < heap_size and arr[right] > arr[largest]:
            largest = right
        if largest != i:
            temp = arr[i]
            arr[i] = arr[largest]
            arr[largest] = temp
            self.max_adus_heapify(arr,largest,heap_size)

    '''
    至底向上构建最大堆，因为i>arr.length/2下标出的结点都为叶结点，所以可以从arr.length/2开始
    '''
    def build_max_heapify(self,arr):
        for i in range(len(arr)-1,-1,-1):
            self.max_adus_heapify(arr,i,len(arr))

    '''
    首先构建最大堆，使用最大堆进行堆排序
    将最大堆中根结点（最大值）与堆末尾结点交换,因为交换后的堆可能不是最大堆了，所以得重建堆，但是堆的大小将减少1
    '''
    def heap_sort(self,arr):
        heap_size = len(arr)
        #构建最大堆
        self.build_max_heapify(arr)
        for i in range(len(arr)-1,0,-1):
            temp = arr[i]
            arr[i] = arr[0]
            arr[0] = temp
            heap_size -= 1
            self.max_adus_heapify(arr,0,heap_size)

if __name__ == '__main__':
    arr = [30,45,22,72,9,12,26,80,22,43,1]
    max_heapify = MaxHeapify()
    max_heapify.heap_sort(arr)
    print(arr)

```

时间复杂度为$$O(nlgn)$$,堆排序是不稳定的。

## 优先队列

优先队列有两种形式：最大优先队列和最小优先队列。优先队列 是不同于先进先出队列的另一种队列。每次从队列中取出的是具有最高优先权的元素（最大或最小）。

在一个包含$$n$$个元素的堆中，所有的优先队列操作都可以在$$O(lgn)$$时间内完成。

# 快速排序

快速排序是不稳定的。

快速排序为什么比堆排序快？

堆排序为$$O(nlgn)$$，快速排序为$$\theta(nlgn)$$。

$$O(nlogn)​$$只代表增长量级，同一个量级前面的常数也可以不一样，不同数量下面的实际运算时间也可以不一样。数量非常小的情况下（比如10以下），插入排序等可能会比快速排序更快。快速排序在平均情况下是$$\theta (nlgn)​$$的。排序算法中时间常数最小的，但是最坏情况下会退化到$$\theta n^{2}​$$，而一般快速排序使用的是固定选取的中位数，为了防止有人使用精心构造的数据来攻击排序算法，$$STL​$$中的std::sort等实现都会采取先快速排序，如果发现明显退化迹象则回退到堆排序这样的时间复杂度稳定的排序上

快速排序最坏情况时间复杂度为$$\Theta (n^{2})$$，但是期望时间复杂度（平均）是$$\Theta (nlgn)$$。

快速排序分治过程：

- 分解：数组$$A[p..r]$$被划分为两个(可能为空)子数组$$A[p..q-1]$$和$$[q+1..r]$$,使得$$A[p..q-1]$$中的每一个元素都小于等于$$A[q]$$，而$$A[q]$$也小于等于$$A[q+1..r]$$中的每个元素。
- 解决：通过递归调用快速排序，对子数组$$A[p..q-1]$$和$$A[q+1..r]$$进行排序。
- 合并：因为子数组都是原址排序的，所以不需要合并操作。

```
def quick_sort(arr,p,r):
    if p < r:
        q = partition(arr,p,r)
        quick_sort(arr,p,q-1)
        quick_sort(arr,q+1,r)
'''
i记录的是存放小于arr[r]的起始位置
'''
def partition(arr,p,r):
    x = arr[r]
    i = p 
    for j in range(p,r):
        if arr[j] <= x:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            i = i +1
    temp = arr[i]
    arr[i] = arr[r]
    arr[r] = temp
    return i
```

## 快速排序性能

如果划分是平衡的，那么快速排序算法性能与归并排序一样。如果划分是不平衡的，那么快速排序的性能就接近于插入排序了。

- 最坏情况划分：当划分产生的两个子问题分别包含了$$n-1$$个元素和$$0$$个元素时，快速排序的最坏情况发生了。
- 最好情况划分：在可能的最平衡的划分中，$$PARTITION$$得到的两个子问题的规模都不大于$$n/2$$。快速排序的平均运行时间更接近于其最好情况。

## 快速排序的随机化版本

与始终采用$$A[r]$$作为主元的方法不同，随机抽样是从子数组$$A[p..r]$$中随机选择一个元素作为主元。因为主元元素是随机选取的，我们期望在平均情况下对输入数组的划分是比较均衡的。

```
import random

def random_quick_sort(arr,p,r):
    if p < r:
        q = random_partition(arr,p,r)
        random_quick_sort(arr,p,q-1)
        random_quick_sort(arr,q+1,r)

def random_partition(arr,p,r):
    index = random.randint(p,r)
    temp = arr[index]
    arr[index] = arr[r] 
    arr[r] = temp
    return partition(arr,p,r)

def partition(arr,p,r):
    x = arr[r]
    i = p 
    for j in range(p,r):
        if arr[j] <= x:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            i = i +1
    temp = arr[i]
    arr[i] = arr[r]
    arr[r] = temp
    return i
```

# 线性时间排序

归并排序、插入排序、堆排序、快速排序等在排序的最终结果中，各元素的次序依赖于它们之间的比较。我们把这类排序算法称为比较排序。

在最坏情况下，任何比较排序算法都需要做$$\Omega (nlgn)$$次比较。

堆排序和归并排序都是渐近最优的比较排序算法。

## 计数排序

计数排序假设$$n$$个输入元素中的每一个都是在0到$$k$$区间内的一个整数，其中$$k$$为某个整数。当$$k=O(n)$$时，排序的运行时间为$$\Theta(n)$$。计数排序是稳定的

计数排序的基本思想是：对每一个输入元素$$x$$，确定小于$$x$$的元素个数。利用这一信息，就可以直接把$$x$$放到它在输出数组中的位置上

```
'''
例如排序数组arr = [2,5,3,0,2,3,0,3]
1.首先统计每个数的数量 arr3的长度k是排序数字的max-min+1 为6 [2,0,2,3,0,1]
如果简单的方式可以直接根据数量表输出排序后数组了，但是需要两层for循环
2.高级一点的做法为：算出每个数的最终位置 [2,2,4,7,7,8]
3，遍历原始数组arr,将选中的数放入其对应的最终位置，并将最终位置往前移动一位
arr1:输入数组
arr2:输出数组
'''
def counting_sort(arr1,arr2,k):
    min_num = min(arr1)
    arr3 = [0 for i in range(k)]
    #1
    for i in range(len(arr1)):
        num = arr1[i] - min_num
        arr3[num] = arr3[num]+1
    print(arr3)
    #2
    for i in range(1,k):
        arr3[i] = arr3[i]+arr3[i-1]
    print(arr3)
    #3
    for i in range(len(arr1)-1,-1,-1):
        arr2[arr3[arr1[i]-min_num]-1] = arr1[i]
        arr3[arr1[i]-min_num] = arr3[arr1[i]-min_num] - 1

```

## 基数排序

基数排序是先按最低有效位进行排序。

类似于将整数第一次按最低位排序，将排序后的又按第二位排序，依次到达最高位

对整数进行排序时，分别将按从低到高位开始，将数字放入桶中，然后再读出

```
'''
从低位到高位对数的每列排序
'''
def radix_sort(arr,radix=10):
    max_num = max(arr)
    for i in range(len(str(max_num))):
    	bucket = [[] for i in range(radix)]
    	print(arr)
    	for value in arr:
    		print(value)
    		print(value%(radix ** (i+1))//(radix**i))
    		bucket[value%(radix ** (i+1))//(radix**i)].append(value)
    	print(bucket)
    	del arr[:]
    	for buc in bucket:
    		arr.extend(buc)
```



## 桶排序

桶排序是将待排序集合中处于同一个值域的元素存入同一个桶中，也就是根据元素值特性将集合拆分为多个区域，则拆分后形成的多个桶，从值域上看是处于有序状态的。对每个桶中元素进行排序，则所有桶中元素构成的集合是已排序的。

快速排序是将集合拆分为两个值域，这里称为两个桶，再分别对两个桶进行排序，最终完成排序。桶排序则是将集合拆分为多个桶，对每个桶进行排序，则完成排序过程。两者不同之处在于，快排是在集合本身上进行排序，属于原地排序方式，且对每个桶的排序方式也是快排。桶排序则是提供了额外的操作空间，在额外空间上对桶进行排序，避免了构成桶过程的元素比较和交换操作，同时可以自主选择恰当的排序算法对桶进行排序。

当然桶排序更是对计数排序的改进，计数排序申请的额外空间跨度从最小元素值到最大元素值，若待排序集合中元素不是依次递增的，则必然有空间浪费情况。桶排序则是弱化了这种浪费情况，将最小值到最大值之间的每一个位置申请空间，更新为最小值到最大值之间每一个固定区域申请空间，尽量减少了元素值大小不连续情况下的空间浪费情况。

**特殊情况**：如果排序100个100以内的数，使用101个桶，分别表示0-100。类似于计数排序。

### 桶排序过程中存在两个关键环节：

- 元素值域的划分，也就是元素到桶的映射规则。映射规则需要根据待排序集合的元素分布特性进行选择，若规则设计的过于模糊、宽泛，则可能导致待排序集合中所有元素全部映射到一个桶上，则桶排序向比较性质排序算法演变。若映射规则设计的过于具体、严苛，则可能导致待排序集合中每一个元素值映射到一个桶上，则桶排序向计数排序方式演化。

- 排序算法的选择，从待排序集合中元素映射到各个桶上的过程，并不存在元素的比较和交换操作，在对各个桶中元素进行排序时，可以自主选择合适的排序算法，桶排序算法的复杂度和稳定性，都根据选择的排序算法不同而不同。

### 算法过程

1. 根据待排序集合中最大元素和最小元素的差值范围和映射规则，确定申请的桶个数；
2. 遍历待排序集合，将每一个元素移动到对应的桶中；
3. 对每一个桶中元素进行排序，并移动到已排序集合中。

```
def insertion_sort(arr,descent):
    for j in range(1,len(arr)):
        key = arr[j]
        i = j -1
        if descent:
            while i >= 0 and arr[i] < key:
                arr[i+1] = arr[i]
                i = i - 1
        else:
            while i >= 0 and arr[i] > key:
                arr[i+1] = arr[i]
                i = i - 1
        arr[i+1] = key

def bucketSort(arr):
    maximum, minimum = max(arr), min(arr)
    bucketArr = [[] for i in range(maximum // 10 - minimum // 10 + 1)]  
    for i in arr:
        index = i // 10 - minimum // 10
        bucketArr[index].append(i)
    arr.clear()
    for i in bucketArr:
        insertion_sort(i,False)
        arr.extend(i)
```



# 中位数和顺序统计量

第$$i$$个顺序统计量是该集合中第$$i$$小的元素。最小值是第1个顺序统计量，最大值是第$$n$$个顺序统计量。

本章将讨论从一个由$$n$$个互异的元素构成的集合中选择第$$i$$个顺序统计量的问题。

## 最小值和最大值

同时找到最小值和最大值，下面有两种方式，第二种方式更优：

- 分别独立的找出最小值和最大值，这各需要$$n-1$$次比较，共需$$2n-2$$次比较。
- 记录已知的最小值和最大值，但我们并不是将每一个输入元素与当前的最小值和最大值进行比较，而是对输入元素成对的进行处理。首先，我们将一对输入进行相互比较，然后将较小的与当前最小值进行比较，把较大的与当前最大值进行比较。这样，对每两个元素需$$3$$次比较，所以总共需要$$3\left \lfloor \frac{n}{2} \right \rfloor$$ 次比较。

## 返回数组第$i$小的元素

### 期望为线性时间的选择算法

使用快速排序中的$$PARTITION$$。同时也可以像快速排序一样，在$$PARTITION$$中加入随机数生成器。

$$RANDOMIZED-SELECT$$的期望运行时间为$$\Theta (n)​$$。

```
import random

def random_partition(arr,p,r):
    index = random.randint(p,r)
    temp = arr[index]
    arr[index] = arr[r] 
    arr[r] = temp
    return partition(arr,p,r)

def partition(arr,p,r):
    x = arr[r]
    i = p 
    for j in range(p,r):
        if arr[j] <= x:
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
            i = i +1
    temp = arr[i]
    arr[i] = arr[r]
    arr[r] = temp
    return i

def randomized_select(arr,p,r,i):
    if p == r:
        return arr[p]
    q = random_partition(arr,p,r)
    k = q-p+1
    if i == k:
        return arr[q]
    elif i < k:
        return randomized_select(arr,p,q-1,i)
    else:
        return randomized_select(arr,q+1,r,i-k)
```

### 最坏情况为线性时间的选择算法

最坏情况运行时间为$$\theta(n)$$的选择算法。

像$$RANDOMIZED-SELECT$$一样，$$SELECT$$算法通过对输入数组的递归划分来找出所需元素，但是在该算法中能够保证得到对数组的一个好的划分。$$SELECT$$使用的也是来自快速排序的确定性划分算法$$PARTITION$$，但做了修改，把划分的主元也作为输入参数。

通过执行下列步骤，算法$$SELECT$$可以确定一个有$$n>1$$个不同元素的输入数组中第$$i$$小的元素。

1. 将输入数组的$$n$$个元素划分为$$\left \lfloor n/5 \right \rfloor$$组，每组$$5$$个元素，且至多只有一组由剩下的$$n mod 5$$个元素组成。
2. 寻找这$$\left \lceil n/5 \right \rceil​$$组中每一组的中位数：首先对每组元素进行插入排序，然后确定每组有序元素的中位数。
3. 对第$$2$$步中找出的$$\left \lceil n/5 \right \rceil$$个中位数，递归调用$$SELECT$$以找出其中位数$$x$$
4. 利用修改过的$$PARTITION$$版本，按中位数的中位数$$x$$对输入数组进行划分。让$$k$$比划分的低区中的元素数目多$$1$$，因此$$x$$是第$$k$$小的元素，并且有$$n-k$$个元素在划分的高区。
5. 如果$$i=k​$$，则返回$$x​$$。如果$$i<k​$$，则在低区递归调用$$SELECT​$$来找出第$$i​$$小的元素。如果$$i>k​$$，则在高区递归查找第$$i-k​$$小的元素。

# 基本数据结构

栈(Stack)实现是**后进先出**($$last-in,first-out,LIFO$$)策略。

队列(queue)的实现是**先进先出**$$（first-in,first-out,FIFO）$$策略。

双端队列(deque，全名double-ended queue)，是一种具有队列和栈性质的数据结构。

链表(linked list)中的各对象按线性顺序排列。数组的线性顺序是由数组下标决定的，然而与数组不同的是，链表的顺序是由各个对象里的指针决定的。

## 指针和对象的实现

有些语言不支持指针和对象数据类型，以下有两种在没有显式的指针数据类型的情况下实现链式数据结构。将利用数组和数组下标来构造对象和指针。

### 对象的多数组表示

对每个属性使用一个数组表示，可以来表示一组有相同属性的对象。

### 对象的单数组表示

计算机内存的字往往从整数$$0$$到$$M-1$$进行编址，其中$$M$$是一个足够大的整数。在许多程序设计语言中，一个对象在计算机内存中占据一组连续的存储单元。指针仅仅是该对象所在的第一个存储单元的地址，要访问其他存储单元可以在指针上加上一个偏移量。在不支持显式的指针数据类型的编程环境下，我们可以采用同样的策略来实现对象。

## 有树根表示

对任意$$n$$个结点的有树根，只需要$$O(n)$$的存储空间。这种左孩子右兄弟表示法：每个结点都包含一个父节点指针$$p$$，且$$T.root$$指向树$$T$$的根结点。然而，每个结点中不是包含指向每个孩子的指针，而是只有两个指针：

1. $$x.left-child$$指向结点$$x$$最左边的孩子结点。
2. $$x.right-sibling$$指向$$x$$右侧相邻的兄弟结点。

如果结点$$x$$没有孩子结点，则$$x.left-child=NIL$$;如果结点$$x$$是其父节点的最右孩子，则$$x.right-sibling=NIL​$$。

![](.\image\左孩子右兄弟表示法.png)

# 散列表(hash table)

## 直接寻址表

为表示动态集合，我们用一个数组，或称为**直接寻址表**，记为$$T[0..m-1]$$。其中每个位置，或称为槽，对应全域$$U$$中的一个关键字。槽$$k$$指向集合中一个关键字为$$k​$$的元素。

![](.\image\直接寻址表.png)

## 散列表

### 解决冲突

- 通过链接法解决冲突
- 开放寻址法

## 散列函数

- 除法散列法
- 乘法散列法
- 全域散列法：随机选择散列函数，使之独立于要存储的关键字。防止将$$n$$个关键字

## 开放寻址法

开放寻址法的好处就在于它不用指针，而是计算出要存取的槽序列。于是，不用存储指针而节省的空间，使得可以用同样的空间来提供更多的槽，潜在地减少了冲突，提高了检索速度。

根据散列函数定位到关键字处，如果该位置有元素，则依次往后查找，如果找到最后还没位置，则需要扩容。扩容不会只增加一个位置，而是增加现有位置的0.72倍，0.72叫做装填因子。

## 完全散列

# 二叉搜索树

## 什么是二叉搜索树

二叉树是每个结点最多有两个子树的结构。

二叉搜索树(二叉查找树，二叉排序树),它或者是一颗空树，或者是具有下列性质的二叉树：若它的左子树不空，则左子树上所有的结点的值均小于它的根结点的值；若它的右子树不空，则右子树上所有结点的值均不大于它的根结点的值；它的左、右子树也分别为二叉搜索树。

二叉搜索树上的基本操作所花费的时间与这棵树的高度成正比。对于有$$n$$个结点的一颗**完全二叉树**来说，这些操作的最坏运行时间为$$\theta(lgn)$$。然而，如果这棵树是一条$$n$$个结点组成的线性链，那么同样的操作就要花费$$\theta(n)$$的最坏运行时间。

![](.\image\二叉搜索树.png)

**中序遍历**：输出的子树根的关键字位于其左子树的关键字值和右子树的关键字值之间。

255678

```
INORDER-TREE-WALK(x):
    if x != None:
        INORDER-TREE-WALK(x.left)
        print(x.key)
        INORDER-TREE-WALK(x.right)
```

**先序遍历**：输出的根的关键字在其左右子树的关键字值之前。

652578

**后序遍历**：输出的根的关键字在其左右子树的关键字值之后。

255876

## 查询二叉搜索树

```
TREE-SEARCH(x,k):
    if x == None or k == x.key:
        return x
    if k < x.key:
        return TREE-SEARCH(x.left,k)
    else:
        return TREE-SEARCH(x.right,k)
```

**最大关键字元素和最小关键字元素**

```
TREE-MINIMUM(x):
    while x.left != None:
        x = x.left
    return x
```

```
TREE-MAXIMUM(x):
    while x.right != None:
        x = x.right
    return x
```

**后继**：一个结点的后继，是大于该结点关键字的最小关键字的结点。

**前驱**：一个结点的前驱，是小于该结点关键字的最大关键字的结点。

搜索二叉树，中序遍历，输出就是升序的排列，比如说1,3,4,5,7,9。一个树结点的后继和前驱就是在这个顺序中，它后面和前面的一个，比如4的后继和前驱就是5,3

```
TREE-SUCCESSOR(x):
    if x.right != None:
        return TREE-MINIMUM(x.right)
    y = x.p
    while y != None and x == y.right:
        x = y
        y = y.p
    return y
```

寻找后继把$$TREE-SUCCESSOR$$分为两种情况：1.如果结点$$x$$的右子树非空，那么$$x$$的后继恰是$$x$$右子树中的最左结点。2,。如果结点$$x$$的右子树为空，那么只需从$$x$$开始沿树而上寻找，如果$$y$$为$$x$$双亲结点并且$$x$$为左结点，那么y即为后继结点，如果$$x$$为右节点，那么得继续往上寻找。比如13的后继为15。

![](.\image\寻找后继.png)

在一棵高度为$$h$$的二叉搜索树上，动态集合上的操作$$SEARCH$$、$$MINIMUM$$、$$MAXIMUM$$、$$SUCCESSOR$$和$$PREDECESSOR$$可以在$$O(h)$$时间内完成。

## 插入和删除

插入：从根结点依次比较向下

删除分为三种情况：

- 如果被删除结点没有孩子结点，那么只是简单地将它删除，并修改它的父节点，用$$NIL$$作为孩子来替换$$z$$。
- 如果被删除结点只有一个孩子，那么将这个孩子提升到树被删除结点的位置上，并修改被删除结点的父节点，用被删除结点的孩子来替换被删除结点。
- 如果被删除结点$$z$$有两个孩子，那么寻找$$z$$的后继替代$$z$$的位置，当后继结点为$$z$$的右孩子时，用后继结点替代$$z$$。如果后继结点不为右孩子时，用后继替代$$z$$位置，并作相应转换。详细内容看算法导论。

在一棵高度为$$h$$的二叉搜索树上，实现动态集合操作$$INSERT$$和$$DELETE$$的运行时间均为$$O(h)$$。

# 红黑树

红黑树是一棵二叉搜索树，它在每个结点上增加了一个存储为来表示结点的颜色，可以是$$RED$$或$$BLACK$$。通过对任何一条从根到叶子的简单路径上各个结点的颜色进行约束，红黑树确保没有一条路径会比其他路径长处$$2$$倍，因而是近似于平衡的。

一棵红黑树是满足下面红黑性质的二叉搜索树：

- 每个结点或是红色的，或是黑色的。
- 根结点是黑色的。
- 每个叶结点$$NIL$$是黑色的。
- 如果一个结点是红色的，则它的两个子结点都是黑色的。
- 对每个结点，从该结点到其所有后代叶结点的简单路径上，均包含相同数目的黑色结点。

# 数据结构的扩张

