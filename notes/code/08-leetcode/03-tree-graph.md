# LeetCode：二叉树 / 图 / 回溯 / 贪心（8 道）

> C++ 题解，全部编译跑通并做了随机对拍。返回 [39 题总表](00-map.md)。
> ⭐ = 两份高频题单的交集，优先级最高。

## 结构定义

```cpp
struct TreeNode {
    int val; TreeNode *left, *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* l, TreeNode* r) : val(x), left(l), right(r) {}
};
```

---

## 二叉树 BFS

### LC102. 二叉树的层序遍历

**思路**：BFS，进循环先把 `q.size()` 存下来当本层节点数。

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) return res;
        queue<TreeNode*> q; q.push(root);
        while (!q.empty()) {
            int sz = q.size();               // 先固定住本层节点数
            vector<int> level;
            for (int i = 0; i < sz; ++i) {
                TreeNode* n = q.front(); q.pop();
                level.push_back(n->val);
                if (n->left)  q.push(n->left);
                if (n->right) q.push(n->right);
            }
            res.push_back(move(level));
        }
        return res;
    }
};
```

**复杂度**：O(n)

**易错**：不存 size 直接循环队列会把下一层也吃进来。

### LC103. 二叉树的锯齿形层序遍历

**思路**：和 102 一样，只是按 `leftToRight` 决定往 `level[i]` 还是 `level[sz-1-i]` 填。

```cpp
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) return res;
        queue<TreeNode*> q; q.push(root);
        bool leftToRight = true;
        while (!q.empty()) {
            int sz = q.size();
            vector<int> level(sz);
            for (int i = 0; i < sz; ++i) {
                TreeNode* n = q.front(); q.pop();
                level[leftToRight ? i : sz - 1 - i] = n->val;  // 直接按位置填，省掉 reverse
                if (n->left)  q.push(n->left);
                if (n->right) q.push(n->right);
            }
            res.push_back(move(level));
            leftToRight = !leftToRight;
        }
        return res;
    }
};
```

**复杂度**：O(n)

**易错**：直接按下标填比先 push 再 reverse 快；奇偶层标志每层翻转。

### LC199. 二叉树的右视图 ⭐

**思路**：BFS，每层只取 `i == sz-1` 那个。

```cpp
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> res;
        if (!root) return res;
        queue<TreeNode*> q; q.push(root);
        while (!q.empty()) {
            int sz = q.size();
            for (int i = 0; i < sz; ++i) {
                TreeNode* n = q.front(); q.pop();
                if (i == sz - 1) res.push_back(n->val);   // 每层最后一个
                if (n->left)  q.push(n->left);
                if (n->right) q.push(n->right);
            }
        }
        return res;
    }
};
```

**复杂度**：O(n)

**易错**：不能只沿右子树走，右子树可能比左子树浅（`[1,2,null,3]` 的右视图是 `1,2,3`）。

## 二叉树 DFS

### LC236. 二叉树的最近公共祖先

**思路**：后序递归：左右都非空则当前是 LCA，否则把非空那个往上传。

```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) return root;
        TreeNode* l = lowestCommonAncestor(root->left,  p, q);
        TreeNode* r = lowestCommonAncestor(root->right, p, q);
        if (l && r) return root;      // 左右各找到一个，当前就是 LCA
        return l ? l : r;             // 否则把找到的那个往上传
    }
};
```

**复杂度**：O(n)

**易错**：`root == p || root == q` 时直接返回 root，覆盖了「一个是另一个祖先」的情况。

## 图搜索

### LC200. 岛屿数量 ⭐

**思路**：遍历网格，遇到 '1' 计数并 DFS 把整块沉掉。

```cpp
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        if (grid.empty()) return 0;
        int m = grid.size(), n = grid[0].size(), ans = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (grid[i][j] == '1') { ++ans; dfs(grid, i, j, m, n); }
        return ans;
    }
private:
    void dfs(vector<vector<char>>& g, int i, int j, int m, int n) {
        if (i < 0 || i >= m || j < 0 || j >= n || g[i][j] != '1') return;
        g[i][j] = '0';                        // 直接改原图当 visited，省一个数组
        dfs(g, i+1, j, m, n); dfs(g, i-1, j, m, n);
        dfs(g, i, j+1, m, n); dfs(g, i, j-1, m, n);
    }
};
```

**复杂度**：O(m·n)

**易错**：直接把 '1' 改成 '0' 当 visited，省一个数组；DFS 四个方向别漏，边界判断放在函数开头统一处理。

## 回溯

### LC46. 全排列

**思路**：**交换法**回溯：`swap(nums[start], nums[i])` 做选择，递归完换回来。

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        backtrack(nums, 0, res);
        return res;
    }
private:
    void backtrack(vector<int>& nums, int start, vector<vector<int>>& res) {
        if (start == (int)nums.size()) { res.push_back(nums); return; }
        for (int i = start; i < (int)nums.size(); ++i) {
            swap(nums[start], nums[i]);              // 做选择
            backtrack(nums, start + 1, res);
            swap(nums[start], nums[i]);              // 撤销选择
        }
    }
};
```

**复杂度**：O(n·n!)

**易错**：交换法不需要 used 数组；`start == n` 时收集答案。有重复元素时要改用 used + 排序去重。

### LC22. 括号生成

**思路**：回溯 + 剪枝：左括号数 `< n` 才能放 `(`，右括号数 `< 左括号数` 才能放 `)`。

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res; string cur;
        backtrack(cur, 0, 0, n, res);
        return res;
    }
private:
    void backtrack(string& cur, int open, int close, int n, vector<string>& res) {
        if ((int)cur.size() == 2 * n) { res.push_back(cur); return; }
        if (open < n)     { cur.push_back('('); backtrack(cur, open+1, close, n, res); cur.pop_back(); }
        if (close < open) { cur.push_back(')'); backtrack(cur, open, close+1, n, res); cur.pop_back(); }
    }
};
```

**复杂度**：O(4ⁿ/√n)

**易错**：剪枝条件是 `close < open` 不是 `close < n`；结果个数是卡特兰数。

## 贪心

### LC55. 跳跃游戏

**思路**：贪心维护 `maxReach`。

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int maxReach = 0;
        for (int i = 0; i < (int)nums.size(); ++i) {
            if (i > maxReach) return false;               // 走不到 i 了
            maxReach = max(maxReach, i + nums[i]);
            if (maxReach >= (int)nums.size() - 1) return true;
        }
        return true;
    }
};
```

**复杂度**：O(n)

**易错**：`i > maxReach` 说明走不到 i，直接 false；判到达终点可以提前 return。
