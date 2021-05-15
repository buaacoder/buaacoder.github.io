---
title: STL容器的总结
date: 2019-09-29 14:58:49
tags:
    - STL
---

# 什么是 STL

STL 是 standard template library 的简写，是标准模板库
STL 里面有许多容器和函数，可以让我们快速的写出一些的数据结构或者实现一些功能
STL 真的太棒了~~~
在此篇文章里，我记录了部分容器的使用方法
对于函数的使用，将会在下一篇中记录

<!--more-->
## STL -- vector

### 解释

vector 是一个数组的模板

### 用法

{% codeblock lang:JavaScript %}
vector <int> v;
vector <int> v1(v);
v.push_back(value); //在尾部加入一个数据
v.pop_back();  //删除最后一个数据
v.clear();     //清除容器中所有数据
v.empty();     //判断容器是否为空
v.size();      //返回容器中实际数据的个数
v[a].swap(v[b])//交换元素
v.begin();     //返回第一个元素的地址
v.front();     //返回第一个元素的值
v.end();       //返回最后一个元素的地址
v.back();      //返回最后一个元素的值
v.erase(pos)  v.erase(begin,end)
v.insert(pos,value)
v.insert(pos,n,value)
{% endcodeblock %}

## STL -- queue/priority_queue/stack

### 解释

queue 是一个队列的模板
priority_queue 是一个优先队列的模板
stack 是一个栈的模板

### 用法

{% codeblock lang:JavaScript %}
queue <int> q;  stack <int> q;
q.push(x);
q.pop();
q.top();
q.front();
q.back();
q.empty();
q.size();
{% endcodeblock %}

### 优先队列 -- priority_queue

{% codeblock lang:JavaScript %}
priority_queue <int> q;
priority_queue <int,vector <int>，greater <int>> q;
用法同queue
{% endcodeblock %}

## STL -- list

### 解释

list 是一个双向链表的模板

### 用法

{% codeblock lang:JavaScript %}
 begin()和end()
 front()和back()
 push_back() 和push_front()
 empty()
 clear()
 insert(pos,num)
 erase(pos)
 sort()                  //将链表排序，默认升序
 remove(num)             //删除链表中匹配num的元素。
 reverse()               // 逆置list
 merge()   l1.merge(l2)  //合并两个链表，合并后l1拥有l1和l2的元素，默认升序排列
{% endcodeblock %}

## STL -- set

### 解释

set 是一个红黑树（一种平衡树）的模板，自带去重效果

### 用法

{% codeblock lang:JavaScript %}
begin()     　　 //返回set容器的第一个元素
end() 　　　　 //返回set容器的最后一个元素
clear()   　　     //删除set容器中的所有的元素
empty() 　　　//判断set容器是否为空
max_size() 　 //返回set容器可能包含的元素最大个数
size() 　　　　//返回当前set容器中的元素个数
find(x)        //返回x的地址，若没有则返回end()
{% endcodeblock %}

## STL -- map

### 解释

map 提供的是一种键值对容器，里面的数据都是成对出现的, 每一对中的第一个值称之为关键字(key)，每个关键字只能在map中出现一次；第二个称之为该关键字的对应值。

### 方法

{% codeblock lang:JavaScript %}
map <int, string> ID_Name;  // 即一个 ID 对应一个名字，其中 ID 为 int 类型，名字为 string 类型

// 使用{}赋值是从c++11开始的，因此编译器版本过低时会报错，如visual studio 2012
map <int, string> ID_Name = {
                { 2015, "Jim" },
                { 2016, "Tom" },
                { 2017, "Bob" } };
插入：
    使用[ ]进行单个插入，ID_Name[2015] = "Tom";  // 如果已经存在键值2015，则会作赋值修改操作，如果没有则插入（2015不是数组下标
    // 插入单个值
    mymap.insert(std::pair<char, int>('a', 100));
    // 列表形式插入
    anothermap.insert({ { 'd', 100 }, {'e', 200} });
取值：
    Map中元素取值主要有at和[ ]两种操作，at会作下标检查，而[]不会。
容量查询：
    // 查询map是否为空
    bool empty();

    // 查询map中键值对的数量
    size_t size();

    // 查询map所能包含的最大键值对数量，和系统和应用库有关。
    // 此外，这并不意味着用户一定可以存这么多，很可能还没达到就已经开辟内存失败了
    size_t max_size();

    // 查询关键字为key的元素的个数，在map里结果非0即1
    size_t count( const Key& key ) const;       // 例：map.count("a")
删除：
    // 删除迭代器指向位置的键值对，并返回一个指向下一元素的迭代器
    iterator erase( iterator pos )

    // 删除一定范围内的元素，并返回一个指向下一元素的迭代器
    iterator erase( const_iterator first, const_iterator last );

    // 根据Key来进行删除， 返回删除的元素数量，在map里结果非0即1
    size_t erase( const key_type& key );

    // 清空map，清空后的size为0
    void clear();
查找：
    // 关键字查询，找到则返回指向该关键字的迭代器，否则返回指向end的迭代器
    // 根据map的类型，返回的迭代器为 iterator 或者 const_iterator
    iterator find (const key_type& k);
    const_iterator find (const key_type& k) const;
{% endcodeblock %}

## STL -- pair

### 解释

pair 是一个储存键值对的容器

### 用法

{% codeblock lang:JavaScript %}
pair <string,double> product1 ("tomatoes",3.25);
pair <string,double> product2;
pair <string,double> product3;
 
product2.first ="lightbulbs"; // type of first is string
product2.second =0.99; // type of second is double
 
product3 = make_pair ("shoes",20.0);
{% endcodeblock %}

## STL -- iterator

### 解释

iterator 是迭代器    可以用来接收容器的地址，如 begin(),end() 等的返回值

### 用法

{% codeblock lang:JavaScript %}
vector <int> v;
vector <int>::iterator it;
while(it = v.begin(); it != v.end(); it++)
{
    //进行操作
}
{% endcodeblock %}

## 特殊说明

若在容器中存放结构，例如：
struct edge{
    int v,w;
}e;
应该如此：vector <edge> v;   注意尖括号内应为结构原来的名称

此外，若容器为 set 容器，则结构中必须重载小于号，若 set <edge,greater <int> >，则要重载大于号
set 中只能有两个参数，vector <int>和 greater <int> 不能同时写进去，而 priority_queue 可以