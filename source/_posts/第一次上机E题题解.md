---
title: 第一次上机E题题解
date: 2019-10-11 12:01:24
tags:
    - 题解
---

## 题目：点灯

### 题目描述
***
有n个灯，编号0∼n−1，一开始都是关闭状态。

每次操作会拨动一个区间[L,R]灯的开关，也就是说，对于灯i，L≤i≤R，如果i是关闭状态，则操作会使灯亮，反之会使灯灭。

请问k次操作后有多少灯亮着。

<!--more-->
### 输入
***
多组输入数据

每组数据第一行两个数n,k（1≤n≤109,1≤k≤105）

接下来k行，每行两个数l,r（0≤l≤r≤n−1）

### 输出
***
每组数据一行一个数，表示最后灯亮的个数

### 输入样例
***
{% codeblock lang:JavaScript %}
10 1
2 6
{% endcodeblock %}

### 输出样例
***
{% codeblock lang:JavaScript %}
5
{% endcodeblock %}

### 思路
***

对于每一个灯，如果它位于一个 [L,R] 区间内，说明开关被按动一次，设这个灯位于 k 个 [L,R] 的区间之内，那么 k 是奇数代表开关被按了奇数次，此时灯是亮的，若 k 是偶数，则灯是灭的。那么问题的关键就是如何求出每个灯的 k 。

因此，读入了 k 个 L 和 k 个 R 之后，我们将这 2*k 个数（ k 个 L 和 k 个 R+1 ）放在一起，并标记每个数是 L 还是 R+1（二维数组和结构数组均可，我使用的是二维数组），由小到大排序，然后定义一个变量 turn = 0（turn 就相当于之前的 k) ，之后对排序后的数组从前向后循环，如果遇到 L ，则 turn += 1 ,说明进入到了一个开关范围内，如果遇到 R+1 ，则 turn -= 1 ，说明离开了一个开关的范围，turn 每次改变后进行判断，若改变后 turn 为奇数，则改变前为偶数，说明从现在这个 L 或 R+1 的位置到上一个 L 或 R+1 的位置中的所有灯的 turn 都是偶数，即灯是灭的，若改变后 turn 为偶数 ，说明从现在的位置到上一个位置的灯都是亮的，那么 ans += 两个位置的差，循环结束即可得到正确结果。

### 代码

{% codeblock lang:JavaScript %}
/*
 Author: 王振
 Result: AC	Submission_id: 1860544
 Created at: Thu Oct 10 2019 17:08:32 GMT+0800 (CST)
 Problem_id: 2489	Time: 824	Memory: 7180
*/
 
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
int l[100005];
int r[100005];
int t[200005][2];
int turn;
int cmp(const void *a, const void *b) { return *(int *)a - *(int *)b; }
int main()
{
    int n, k;
    while (cin >> n >> k)
    {
        int i;
        int p = 0;
        for (i = 1; i <= k; i++)
        {
            scanf("%d%d", &l[i], &r[i]);
            t[++p][0] = l[i];
            t[p][1] = 1;
            t[++p][0] = r[i] + 1;
            t[p][1] = 2;
        }
        qsort(t + 1, k * 2, 8, cmp);
        int ans = 0;
        for (i = 1; i <= k * 2; i++)
        {
            if (t[i][1] == 1)
            {
                turn++;
            }
            else
            {
                turn--;
            }
            if (turn % 2 == 0)
            {
                ans += t[i][0] - t[i - 1][0];
            }
        }
        cout << ans << endl;
    }
    return 0;
}
{% endcodeblock %}