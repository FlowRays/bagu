# LeetCode：链表 / 栈 / 堆 / 区间（9 道）

> C++ 题解，全部编译跑通并做了随机对拍。返回 [39 题总表](00-map.md)。
> ⭐ = 两份高频题单的交集，优先级最高。

## 结构定义

```cpp
struct ListNode {
    int val; ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *n) : val(x), next(n) {}
};
```

---

## 链表基础

### LC206. 反转链表

**思路**：三指针 prev/cur/next 逐个反转。

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *prev = nullptr, *cur = head;
        while (cur) {
            ListNode* nxt = cur->next;   // 先存住后继，否则改完指针就找不到了
            cur->next = prev;
            prev = cur; cur = nxt;
        }
        return prev;
    }
};
```

**复杂度**：O(n) / O(1)

**易错**：改 `cur->next` 之前必须先存住后继。

### LC21. 合并两个有序链表

**思路**：哨兵节点 + 尾插。

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* a, ListNode* b) {
        ListNode dummy, *tail = &dummy;      // 哨兵节点省掉头节点特判
        while (a && b) {
            if (a->val <= b->val) { tail->next = a; a = a->next; }
            else                  { tail->next = b; b = b->next; }
            tail = tail->next;
        }
        tail->next = a ? a : b;
        return dummy.next;
    }
};
```

**复杂度**：O(m+n)

**易错**：用 `ListNode dummy;` 栈上对象省一次 new；最后 `tail->next = a ? a : b` 一次接完剩余。

### LC19. 删除链表的倒数第 N 个节点

**思路**：快指针先走 n 步，然后同步走到尾。

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0, head);             // 删的可能是头节点，必须用哨兵
        ListNode *fast = &dummy, *slow = &dummy;
        while (n--) fast = fast->next;       // fast 先走 n 步
        while (fast->next) { fast = fast->next; slow = slow->next; }
        ListNode* del = slow->next;
        slow->next = del->next;
        delete del;
        return dummy.next;
    }
};
```

**复杂度**：O(n) / O(1)

**易错**：**必须用哨兵**，否则删头节点要特判；快指针从 dummy 出发走 n 步，循环条件是 `fast->next`。

## 链表进阶

### LC25. K 个一组翻转链表 ⭐

**思路**：每组先找到第 k 个节点，找不到就停；组内反转时把 `prev` 初始化成 `groupNext`，反转完自动接上。

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode dummy(0, head), *groupPrev = &dummy;
        while (true) {
            ListNode* kth = groupPrev;       // 找到本组第 k 个节点
            for (int i = 0; i < k && kth; ++i) kth = kth->next;
            if (!kth) break;                 // 不足 k 个，保持原样
            ListNode* groupNext = kth->next;
            // 反转 [groupPrev->next, kth]，反转后尾巴接 groupNext
            ListNode *prev = groupNext, *cur = groupPrev->next;
            while (cur != groupNext) {
                ListNode* nxt = cur->next;
                cur->next = prev; prev = cur; cur = nxt;
            }
            ListNode* newTail = groupPrev->next;   // 原来的头变成了尾
            groupPrev->next = kth;                 // kth 变成了这组的头
            groupPrev = newTail;
        }
        return dummy.next;
    }
};
```

**复杂度**：O(n) / O(1)

**易错**：记住三个指针 `groupPrev / kth / groupNext`；反转后**原来的头变成尾**，要用它当下一组的 groupPrev。不足 k 个保持原序。

### LC23. 合并 K 个升序链表

**思路**：分治两两归并。也可以用小顶堆。

```cpp
class Solution {
public:
    // 分治：两两归并，O(N log k)
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        return go(lists, 0, (int)lists.size() - 1);
    }
private:
    ListNode* go(vector<ListNode*>& v, int l, int r) {
        if (l == r) return v[l];
        int mid = l + (r - l) / 2;
        return merge2(go(v, l, mid), go(v, mid + 1, r));
    }
    ListNode* merge2(ListNode* a, ListNode* b) {
        ListNode dummy, *tail = &dummy;
        while (a && b) {
            if (a->val <= b->val) { tail->next = a; a = a->next; }
            else                  { tail->next = b; b = b->next; }
            tail = tail->next;
        }
        tail->next = a ? a : b;
        return dummy.next;
    }
};
// 小顶堆解法，同样 O(N log k)
class SolutionHeap {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
        priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);
        for (auto* l : lists) if (l) pq.push(l);
        ListNode dummy, *tail = &dummy;
        while (!pq.empty()) {
            ListNode* t = pq.top(); pq.pop();
            tail->next = t; tail = t;
            if (t->next) pq.push(t->next);
        }
        tail->next = nullptr;
        return dummy.next;
    }
};
```

**复杂度**：O(N log k)

**易错**：分治是 log k 层每层 O(N)；堆解法最后要 `tail->next = nullptr`，否则可能带出脏尾巴。

### LC146. LRU 缓存 ⭐

**思路**：哈希表 + 双向链表，两个哨兵节点。

```cpp
class LRUCache {
    struct Node { int k, v; Node *prev, *next; Node(int k=0,int v=0):k(k),v(v),prev(nullptr),next(nullptr){} };
    int cap;
    unordered_map<int, Node*> mp;
    Node *head, *tail;                       // head 后面是最近用的，tail 前面是最久没用的
    void remove(Node* n) { n->prev->next = n->next; n->next->prev = n->prev; }
    void pushFront(Node* n) {
        n->next = head->next; n->prev = head;
        head->next->prev = n; head->next = n;
    }
public:
    LRUCache(int capacity) : cap(capacity) {
        head = new Node(); tail = new Node();  // 两个哨兵，省掉所有边界判断
        head->next = tail; tail->prev = head;
    }
    ~LRUCache() {
        Node* c = head; while (c) { Node* n = c->next; delete c; c = n; }
    }
    int get(int key) {
        auto it = mp.find(key);
        if (it == mp.end()) return -1;
        remove(it->second); pushFront(it->second);   // 命中就挪到最前
        return it->second->v;
    }
    void put(int key, int value) {
        auto it = mp.find(key);
        if (it != mp.end()) {
            it->second->v = value;
            remove(it->second); pushFront(it->second);
            return;
        }
        if ((int)mp.size() == cap) {                 // 淘汰 tail 前一个
            Node* last = tail->prev;
            remove(last); mp.erase(last->k); delete last;
        }
        Node* n = new Node(key, value);
        mp[key] = n; pushFront(n);
    }
};
```

**复杂度**：O(1)

**易错**：`get` 命中也要挪到头部；`put` 已存在时是更新+挪动**不是**插入；淘汰时别忘了从 map 里 erase。

## 栈

### LC20. 有效的括号 ⭐

**思路**：右括号查表比对栈顶。

```cpp
class Solution {
public:
    bool isValid(string s) {
        stack<char> st;
        unordered_map<char, char> pair{{')','('},{']','['},{'}','{'}};
        for (char c : s) {
            if (pair.count(c)) {                     // 右括号
                if (st.empty() || st.top() != pair[c]) return false;
                st.pop();
            } else st.push(c);
        }
        return st.empty();                           // 结尾必须清空
    }
};
```

**复杂度**：O(n)

**易错**：栈空时遇到右括号直接 false；结尾必须 `st.empty()`。

## 堆与 Top-K

### LC215. 数组中的第 K 个最大元素 ⭐

**思路**：快速选择：第 k 大 = 升序第 `n-k` 小，每次 partition 后只递归一边。

```cpp
// 方法 1：大小为 k 的小顶堆，O(n log k)
class SolutionHeap {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int, vector<int>, greater<int>> pq;
        for (int x : nums) {
            pq.push(x);
            if ((int)pq.size() > k) pq.pop();        // 堆里永远只留最大的 k 个
        }
        return pq.top();
    }
};
// 方法 2：快速选择，平均 O(n)
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int target = (int)nums.size() - k;           // 第 k 大 = 升序第 n-k 小
        int l = 0, r = (int)nums.size() - 1;
        mt19937 rng(random_device{}());
        while (true) {
            int p = partition(nums, l, r, rng);
            if (p == target) return nums[p];
            if (p < target) l = p + 1; else r = p - 1;
        }
    }
private:
    int partition(vector<int>& a, int l, int r, mt19937& rng) {
        swap(a[l + (int)(rng() % (r - l + 1))], a[r]);   // 随机 pivot 防最坏情况
        int pivot = a[r], i = l;
        for (int j = l; j < r; ++j) if (a[j] < pivot) swap(a[i++], a[j]);
        swap(a[i], a[r]);
        return i;
    }
};
```

**复杂度**：平均 O(n)

**易错**：**pivot 必须随机**否则有序数组退化 O(n²)；另一种是大小为 k 的**小顶堆** O(n log k)，堆顶就是答案。

## 区间

### LC56. 合并区间

**思路**：按左端点排序，和结果数组最后一个比较。

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());        // 按左端点排序
        vector<vector<int>> res;
        for (auto& iv : intervals) {
            if (!res.empty() && iv[0] <= res.back()[1])  // 和上一个结果重叠
                res.back()[1] = max(res.back()[1], iv[1]);
            else res.push_back(iv);
        }
        return res;
    }
};
```

**复杂度**：O(n log n)

**易错**：`iv[0] <= res.back()[1]` 用 `<=`（相邻区间 `[1,4],[4,5]` 要合并）；合并时取 `max` 右端点，因为可能被完全包含。
