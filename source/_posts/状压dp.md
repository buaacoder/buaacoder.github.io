---
title: 状压dp
date: 2019-09-22 22:33:50
tags: [算法,动态规划]
---

## 题目：互不侵犯

### 题目描述

在 N*N 的棋盘里面放 K 个国王，使他们互不攻击，共有多少种摆放方案。国王能攻击到它上下左右，以及左上左下右上右下八个方向上附近的各一个格子，共 8 个格子。
<!--more-->
### 输入

只有一行，包含两个数 N, K。

### 输出

所得的方案数。

### 输入样例

{% codeblock  %}
3 2
{% endcodeblock %}

### 输出样例

{% codeblock  %}
16
{% endcodeblock %}

## 思路

### 关于 状压dp

状压dp是动态规划的一种，通过将状态压缩为整数来达到优化转移的目的。
具体来说，我们可以用一个二进制数的每一个二进制位来表示一个位置的状态，在这个题中，我们就可以用 0 来表示该位置不放置国王，用 1 来表示该位置放置国王
因为棋盘是一个 N*N 大小的矩阵，我们就可以每一行用一个二进制数来表示该行国王的放置情况

### 具体操作

见代码注释

### 代码

这里是本蒟蒻的代码~

{% codeblock lang:JavaScript %}
#include <iostream>
#include <algorithm>
using namespace std;

int n,k,cnt;    // n为棋盘的大小，k为国王的个数，cnt为只考虑一行的情况下（即一个国王的左右不能放置国王）放置国王的所有可能情况（国王为任意数量）
long long sta[2005],sit[2005];      // sta数组存储各个情况放置的国王的数目 sit数组存储各个情况下国王的放置位置（用一个二进制数来表示）
int f[15][2005][105];               // f数组的第一维是当前的行数，第二维是放置国王的所有情况中的第几个，第三维是到该行总共放置国王的个数

void dfs(int x,int num,int cur)    // 预处理出单行情况放置国王的所有情况 x:国王的放置位置(一个二进制数) num:放置国王的个数 cur:当前搜到的位置
{
	if(cur>=n)         // cur>=n 表示一行搜完
	{
		sit[++cnt]=x;  
		sta[cnt]=num;
		return;
	}
	dfs(x,num,cur+1);     //该位置不放国王
	dfs(x+(1<<cur),num+1,cur+2);      //该位置放国王
}

int main()
{
	while(cin>>n>>k)
	{
		dfs(0,0,0);        //预处理出所有情况
		for(int i=1;i<=cnt;i++)       //将结果赋给第一行
		{
			f[1][i][sta[i]]=1;
		}
		for(int i=2;i<=n;i++)         //从第 2 行到第 n 行，对于前一行的所有可能状态，当前行用所有的可能状态进行比较
		{
			for(int j=1;j<=cnt;j++)
			{
				for(int l=1;l<=cnt;l++)
				{
					if(sit[j]&sit[l]) continue;
					if((sit[j]<<1)&sit[l]) continue;
					if(sit[j]&(sit[l]<<1)) continue;
                    //上面这三行用来排除不合法的转移     即当前行有国王的上方或左上或右上存在国王

					for(int p=sta[j];p<=k;p++)     // 如果两行没有冲突  则当前行放置 sta[j] 个国王
					{
						f[i][j][p]+=f[i-1][l][p-sta[j]];
					}
				}
			}
		}
		long long ans=0;       // ans为答案
		for(int i=1;i<=cnt;i++)      // ans加上第n行的每一种情况下放置k个国王的总数
		{
			ans+=f[n][i][k];
		}
		cout<<ans<<endl;
	}
	return 0;
}
{% endcodeblock %}

感谢观看~