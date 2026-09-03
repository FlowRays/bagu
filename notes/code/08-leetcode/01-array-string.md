# LeetCode：数组 / 字符串 / 二分（12 道）

> C++ 题解，全部编译跑通并做了随机对拍。返回 [39 题总表](00-map.md)。
> ⭐ = 两份高频题单的交集，优先级最高。

## 数组与哈希

### LC49. 字母异位词分组

**思路**：排序后的字符串当哈希 key。

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (auto& s : strs) {
            string key = s;
            sort(key.begin(), key.end());       // 排序后的串当 key
            mp[key].push_back(s);
        }
        vector<vector<string>> res;
        for (auto& [_, v] : mp) res.push_back(move(v));
        return res;
    }
};
```

**复杂度**：O(n·k log k)

**易错**：key 用 `string` 别用 `vector<char>`；也可以用 26 位计数当 key 做到 O(n·k)。

### LC53. 最大子数组和

**思路**：Kadane：`cur = max(x, cur+x)`，要么接上前面要么从我重开。

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int cur = nums[0], best = nums[0];
        for (size_t i = 1; i < nums.size(); ++i) {
            cur = max(nums[i], cur + nums[i]);  // 要么接上前面，要么从我重开
            best = max(best, cur);
        }
        return best;
    }
};
```

**复杂度**：O(n)

**易错**：`best` 初值必须是 `nums[0]` 不是 0，否则全负数会返回 0。

## 排序与数组操作

### LC88. 合并两个有序数组

**思路**：**从后往前**三指针填。

```cpp
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while (j >= 0) {                        // 从后往前填，避免覆盖未处理的元素
            if (i >= 0 && nums1[i] > nums2[j]) nums1[k--] = nums1[i--];
            else nums1[k--] = nums2[j--];
        }
    }
};
```

**复杂度**：O(m+n)

**易错**：从前往后会覆盖 nums1 里还没处理的元素；循环条件只看 `j >= 0`，j 走完就结束（nums1 剩下的本来就在原位）。

### LC912. 排序数组

**思路**：归并排序。复用一个 buf 避免每层新建 vector。

```cpp
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        vector<int> buf(nums.size());
        msort(nums, buf, 0, (int)nums.size() - 1);
        return nums;
    }
private:
    void msort(vector<int>& a, vector<int>& buf, int l, int r) {
        if (l >= r) return;
        int mid = l + (r - l) / 2;
        msort(a, buf, l, mid);
        msort(a, buf, mid + 1, r);
        int i = l, j = mid + 1, k = l;
        while (i <= mid && j <= r) buf[k++] = a[i] <= a[j] ? a[i++] : a[j++];
        while (i <= mid) buf[k++] = a[i++];
        while (j <= r)   buf[k++] = a[j++];
        for (int t = l; t <= r; ++t) a[t] = buf[t];
    }
};
```

**复杂度**：O(n log n) / O(n)

**易错**：面试要求手写就别调 `sort`；快排要随机 pivot，否则有序输入退化成 O(n²)。

## 双指针

### LC15. 三数之和 ⭐

**思路**：排序 + 固定第一个数 + 左右双指针。

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        vector<vector<int>> res;
        for (int i = 0; i + 2 < n; ++i) {
            if (nums[i] > 0) break;                    // 最小的都 >0 不可能凑成 0
            if (i && nums[i] == nums[i - 1]) continue; // 跳过重复的第一个数
            int l = i + 1, r = n - 1;
            while (l < r) {
                int s = nums[i] + nums[l] + nums[r];
                if (s < 0) ++l;
                else if (s > 0) --r;
                else {
                    res.push_back({nums[i], nums[l], nums[r]});
                    ++l; --r;
                    while (l < r && nums[l] == nums[l - 1]) ++l;   // 去重
                    while (l < r && nums[r] == nums[r + 1]) --r;
                }
            }
        }
        return res;
    }
};
```

**复杂度**：O(n²)

**易错**：**三处去重**：外层跳过 `nums[i]==nums[i-1]`，命中后内层跳过左右重复值；`nums[i] > 0` 可以直接 break。

### LC42. 接雨水 ⭐

**思路**：双指针：谁矮就先结算谁，因为矮的那侧水位已经被自己的 max 锁死。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int l = 0, r = (int)height.size() - 1;
        int lmax = 0, rmax = 0, ans = 0;
        while (l < r) {
            if (height[l] < height[r]) {        // 左边矮 -> 左侧水位由 lmax 决定
                lmax = max(lmax, height[l]);
                ans += lmax - height[l]; ++l;
            } else {
                rmax = max(rmax, height[r]);
                ans += rmax - height[r]; --r;
            }
        }
        return ans;
    }
    // 单调栈解法：横着一层层接
    int trapStack(vector<int>& height) {
        stack<int> st; int ans = 0;
        for (int i = 0; i < (int)height.size(); ++i) {
            while (!st.empty() && height[st.top()] < height[i]) {
                int bottom = st.top(); st.pop();
                if (st.empty()) break;
                int w = i - st.top() - 1;
                ans += w * (min(height[st.top()], height[i]) - height[bottom]);
            }
            st.push(i);
        }
        return ans;
    }
};
```

**复杂度**：O(n) / O(1)

**易错**：两种解法都要会：双指针是竖着按列算，单调栈是**横着一层层**算；栈解法 pop 出 bottom 后栈空要 break。

## 滑动窗口

### LC3. 无重复字符的最长子串 ⭐

**思路**：滑动窗口 + 记录每个字符**上次出现的位置**。

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        vector<int> last(128, -1);
        int left = 0, ans = 0;
        for (int i = 0; i < (int)s.size(); ++i) {
            if (last[s[i]] >= left) left = last[s[i]] + 1;  // 左边界只右移不回退
            last[s[i]] = i;
            ans = max(ans, i - left + 1);
        }
        return ans;
    }
};
```

**复杂度**：O(n)

**易错**：`left` 只能右移不能回退，所以要判 `last[c] >= left`；用 `vector<int>(128,-1)` 比 map 快。

### LC76. 最小覆盖子串

**思路**：滑窗 + `miss` 计数：右扩到覆盖，再尽量收缩左边界。

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        if (s.empty() || t.empty()) return "";
        vector<int> need(128, 0);
        for (char c : t) ++need[c];
        int miss = t.size();                     // 还差多少个字符（含重复）
        int bestLen = INT_MAX, bestL = 0, left = 0;
        for (int right = 0; right < (int)s.size(); ++right) {
            if (need[s[right]]-- > 0) --miss;
            while (miss == 0) {                  // 已覆盖，尽量收缩左边界
                if (right - left + 1 < bestLen) { bestLen = right - left + 1; bestL = left; }
                if (++need[s[left]] > 0) ++miss;
                ++left;
            }
        }
        return bestLen == INT_MAX ? "" : s.substr(bestL, bestLen);
    }
};
```

**复杂度**：O(n)

**易错**：`need` 可以为负（多余字符）；`if (need[s[right]]-- > 0) --miss` 这种写法要看清是后置减；收缩时先更新答案再移动。

## 二分查找

### LC34. 在排序数组中查找元素的第一个和最后一个位置

**思路**：两次 lower_bound：`lower(target)` 和 `lower(target+1)-1`。

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int a = lowerBound(nums, target);
        if (a == (int)nums.size() || nums[a] != target) return {-1, -1};
        return {a, lowerBound(nums, target + 1) - 1};
    }
private:
    int lowerBound(vector<int>& nums, long long x) {   // 第一个 >= x 的下标
        int lo = 0, hi = nums.size();
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] < x) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
};
```

**复杂度**：O(log n)

**易错**：`target+1` 在 target = INT_MAX 时会溢出，参数用 `long long`；先判 `a == n || nums[a] != target`。

### LC33. 搜索旋转排序数组

**思路**：二分时先判断哪半边有序，再看 target 落不落在有序那半的区间里。

```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int lo = 0, hi = (int)nums.size() - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) return mid;
            if (nums[lo] <= nums[mid]) {                 // 左半有序
                if (nums[lo] <= target && target < nums[mid]) hi = mid - 1;
                else lo = mid + 1;
            } else {                                      // 右半有序
                if (nums[mid] < target && target <= nums[hi]) lo = mid + 1;
                else hi = mid - 1;
            }
        }
        return -1;
    }
};
```

**复杂度**：O(log n)

**易错**：判有序用 `nums[lo] <= nums[mid]`（`<=` 不能少，lo==mid 时要成立）；target 的区间判断是**闭开**要对应上。

## 矩阵

### LC48. 旋转图像

**思路**：**转置 + 每行 reverse = 顺时针 90°**。

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n; ++i)                  // 转置
            for (int j = i + 1; j < n; ++j)
                swap(matrix[i][j], matrix[j][i]);
        for (auto& row : matrix)                     // 每行翻转
            reverse(row.begin(), row.end());
    }
};
```

**复杂度**：O(n²) / O(1)

**易错**：转置内层从 `i+1` 开始，从 0 开始等于转两次白干；逆时针是「每行 reverse + 转置」。

### LC54. 螺旋矩阵

**思路**：四个边界 top/bot/left/right 往里收。

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.empty()) return {};
        int top = 0, bot = matrix.size() - 1;
        int left = 0, right = matrix[0].size() - 1;
        vector<int> res;
        while (top <= bot && left <= right) {
            for (int j = left; j <= right; ++j) res.push_back(matrix[top][j]);
            ++top;
            for (int i = top; i <= bot; ++i) res.push_back(matrix[i][right]);
            --right;
            if (top <= bot) {                        // 防止单行被重复扫
                for (int j = right; j >= left; --j) res.push_back(matrix[bot][j]);
                --bot;
            }
            if (left <= right) {                     // 防止单列被重复扫
                for (int i = bot; i >= top; --i) res.push_back(matrix[i][left]);
                ++left;
            }
        }
        return res;
    }
};
```

**复杂度**：O(m·n)

**易错**：**扫第三、第四条边前必须重新判边界**，否则单行或单列会被扫两遍。
