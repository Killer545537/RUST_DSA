/* Dynamic Programming (DP) is a method used to solve problems by dividing them into smaller, easier subproblems.
The two types of approaches to DP, Top-Down(Memoization) and Bottom-Up(Tabulation)
In memoization, we start from the top and recurse down while storing the results to the sub-problems
In tabulation, we start from the bottom and recurse up using the results of the sub-problems  */
use std::collections::HashMap;
use std::sync::Mutex;
use itertools::Itertools;
use rayon::prelude::*;
use crate::stack_queues::largest_rectangle_area_optimised;

///The simplest recursive approach
pub fn fibonacci(n: i32) -> i32 {
    match n {
        0 | 1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2)
    }
}
//This is terrible because each call is computed again and again wasting time
pub fn fibonacci_better(n: i32) -> i32 {
    let n = n as usize;
    let mut dp = vec![-1; n + 1];
    dp[0] = 1;
    dp[1] = 1;

    for i in 2..=n {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    dp[n]
}
//Even this can be optimized, since only the last two values are being used, only use them
pub fn fibonacci_best(n: i32) -> i32 {
    let (mut prev1, mut prev2) = (1, 1);

    for _ in 2..=n {
        let current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }

    prev1
}

///From stair i we can jump to i + 1 or i + 2, find the number of ways to climb n stairs
pub fn climb_stairs(n: i32) -> i32 {
    //The recurrence relation is f(n) = f(n-1) + f(n-2), and f(0) = 0, f(1) = 1, f(2) = 2 ((1,1), (2))
    let (mut prev1, mut prev2) = (1, 1);

    for _ in 2..=n {
        let current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }

    prev1
}

///A frog can jump from heights[i]->heights[i+1]|heights[i+2] using energy |heights[i']-heights[i]|
pub fn frog_jump_recursive(heights: &[i32]) -> i32 {
    if heights.len() == 1 { //If there is only one step, no energy is spent
        return 0;
    }

    if heights.len() == 2 {
        return (heights[1] - heights[0]).abs();
    }

    let one = (heights[1] - heights[0]).abs() + frog_jump_recursive(&heights[1..]);
    let two = (heights[2] - heights[0]).abs() + frog_jump_recursive(&heights[2..]);

    return std::cmp::min(one, two);
}
pub fn frog_jump(heights: &[i32]) -> i32 {
    let mut dp = vec![-1; heights.len()];

    fn helper(heights: &[i32], n: usize, dp: &mut Vec<i32>) -> i32 { //This uses 1-based indexing, so indices are off by 1
        if n == 1 {
            return 0;
        }

        if n == 2 {
            return (heights[0] - heights[1]).abs();
        }

        if dp[n - 1] != -1 {
            return dp[n - 1];
        }

        let one = helper(heights, n - 1, dp) + (heights[n - 1] - heights[n - 2]).abs();
        let two = helper(heights, n - 2, dp) + (heights[n - 1] - heights[n - 3]).abs();

        dp[n - 1] = std::cmp::min(one, two);

        return dp[n - 1];
    }

    helper(heights, heights.len(), &mut dp)
}
pub fn frog_jump_optimised(heights: &[i32]) -> i32 {
    let mut dp = vec![0; heights.len()];

    for i in 1..heights.len() {
        let one = dp[i - 1] + (heights[i] - heights[i - 1]).abs();
        let two = if i > 1 {
            dp[i - 2] + (heights[i] - heights[i - 2]).abs()
        } else {
            i32::MAX
        };

        dp[i] = std::cmp::min(one, two);
    }

    dp[heights.len() - 1]
}
pub fn frog_jump_space_optimised(heights: &[i32]) -> i32 {
    let (mut prev1, mut prev2) = (0, 0);

    for i in 1..heights.len() {
        let one = prev1 + (heights[i] - heights[i - 1]).abs();
        let two = if i > 1 {
            prev2 + (heights[i] - heights[i - 2]).abs()
        } else {
            i32::MAX
        };

        let curr = std::cmp::min(one, two);
        prev2 = prev1;
        prev1 = curr;
    }

    prev1
}

pub fn rob(houses: &[i32]) -> i32 {
    let n = houses.len();
    let mut dp = vec![0; n]; //The maximum take from the first 'i' houses
    dp[0] = houses[0];

    for i in 2..n { //The current house is i
        let take = houses[i] + dp[i - 2];
        let not_take = dp[i - 1];
        dp[i] = std::cmp::max(take, not_take);
    }

    dp[n - 1]
}
pub fn rob_optimised(houses: &[i32]) -> i32 {
    let (mut prev1, mut prev2) = (0, 0);

    for &house in houses {
        let take = house + prev2;
        let not_take = prev1;
        let curr = std::cmp::max(take, not_take);

        prev2 = prev1;
        prev1 = curr;
    }

    prev1
}

///Here, the houses are in a circle (first and last are connected)
pub fn rob_2(houses: &[i32]) -> i32 {
    //Take either the first house or the last house
    std::cmp::max(rob_optimised(&houses[1..]), rob_optimised(&houses[..houses.len() - 1]))
}

//Now we see 2-D dynamic programming
///The training is of n days, each day with three activities, the same activity cannot be done on two consecutive days
pub fn maximum_points(points: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![-1; 3]; points.len()];
    fn helper(points: &Vec<Vec<i32>>, current_day: usize, prev_task: usize, dp: &mut Vec<Vec<i32>>) -> i32 {
        if current_day == points.len() {
            return 0;
        }

        if dp[current_day][prev_task] != -1 {
            return dp[current_day][prev_task];
        }

        let mut max_score = 0;
        for task in 0..3 {
            if task == prev_task {
                continue;
            }
            let score = points[current_day][task] + helper(points, current_day + 1, task, dp);
            max_score = std::cmp::max(max_score, score);
        }
        max_score
    }

    helper(&points, 0, 0, &mut dp)
}
//We can convert the 2-D DP into a 1-D DP since only the scores of the previous day matter
pub fn maximum_points_optimised(points: Vec<Vec<i32>>) -> i32 {
    if points.is_empty() {
        return 0;
    }

    let mut dp = points[0].clone();

    for day in 1..points.len() {
        let mut temp = vec![0; 3];
        for task in 0..3 {
            for prev_task in 0..3 {
                if task != prev_task {
                    temp[task] = std::cmp::max(temp[task], dp[prev_task] + points[day][task]);
                }
            }
        }

        dp = temp;
    }

    *dp.iter().max().unwrap()
}

///The total number of unique paths from (0,0) to (m-1, n-1) moving down or right
pub fn unique_paths_recursive(m: i32, n: i32) -> i32 {
    //The number of ways to reach a cell = (ways to reach right cell) + (ways to reach down cell)
    //On the bottom and right boundary, the number of ways to reach the cell is 1 (ways to reach right or down = 0)
    if m == 1 || n == 1 {
        return 1;
    }

    return unique_paths_recursive(m - 1, n) + unique_paths_recursive(m, n - 1);
}
pub fn unique_paths_dp(m: i32, n: i32) -> i32 {
    let m = m as usize;
    let n = n as usize;
    let mut dp = vec![vec![1; n]; m];
    for i in 1..m {
        for j in 1..n {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }

    dp[m - 1][n - 1]
}

pub fn unique_paths(m: i32, n: i32) -> i32 {
    let m = m as usize;
    let n = n as usize;
    let mut prev = vec![1; n];
    for _ in 1..m {
        let mut curr = vec![1; n];
        for i in 1..n {
            curr[i] = curr[i - 1] + prev[i];
        }
        prev = curr;
    }

    prev[n - 1]
}
//The problem can be solved using Combinatorics. The number of ways = (m+n-2)C(m-1)

///1-> obstacle
pub fn unique_path_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let m = obstacle_grid.len();
    let n = obstacle_grid[0].len();
    if obstacle_grid[0][0] == 1 || obstacle_grid[m - 1][n - 1] == 1 { //If the starting/ending points are blocked, we cannot reach the end
        return 0;
    }
    let mut dp = vec![vec![0; n]; m];
    dp[0][0] = 1;
    for i in 1..m { //First column
        if obstacle_grid[i][0] == 0 { //If it is not a blockage
            dp[i][0] = dp[i - 1][0];
        }
    }
    for j in 1..n { //First row
        if obstacle_grid[0][j] == 0 { //If it is not a blockage
            dp[0][j] = dp[0][j - 1];
        }
    }

    for i in 1..m {
        for j in 1..n {
            if obstacle_grid[i][j] == 0 {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
    }

    dp[m - 1][n - 1]
}
pub fn unique_path_with_obstacles_optimised(obstacle_grid: Vec<Vec<i32>>) -> i32 {
    let m = obstacle_grid.len();
    let n = obstacle_grid[0].len();
    if obstacle_grid[0][0] == 1 || obstacle_grid[m - 1][n - 1] == 1 {
        return 0;
    }
    let mut prev = vec![0; n];
    prev[0] = 1;
    for i in 0..m {
        let mut curr = vec![0; n];
        for j in 0..n {
            if obstacle_grid[i][j] == 1 {
                curr[j] = 0;
            } else if j > 0 {
                curr[j] = curr[j - 1] + prev[j];
            } else { //If it is the first row
                curr[j] = prev[j];
            }
        }
        prev = curr;
    }

    prev[n - 1]
}

///Only move down/right
pub fn min_path_sum_recursive(grid: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    fn helper(grid: &Vec<Vec<i32>>, m: usize, n: usize) -> i32 {
        if m == 0 && n == 0 {
            return grid[m][n];
        }
        if m == 0 { //If we are at the first row, there is no going up
            return grid[m][n] + helper(grid, m, n - 1);
        }
        if n == 0 { //If we are at the first column, there is no going left
            return grid[m][n] + helper(grid, m - 1, n);
        }
        return grid[m][n] + std::cmp::min(helper(grid, m - 1, n), helper(grid, m, n - 1));
    }

    helper(&grid, m - 1, n - 1)
}
pub fn min_path_sum(grid: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    let mut dp = vec![vec![-1; n]; m];
    dp[0][0] = grid[0][0];

    //First row
    for j in 1..n {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }
    //First column
    for i in 1..m {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }

    for i in 1..m {
        for j in 1..n {
            dp[i][j] = std::cmp::min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
        }
    }

    dp[m - 1][n - 1]
}
//Convert to a 1-D DP
pub fn min_path_sum_optimised(grid: Vec<Vec<i32>>) -> i32 {
    let m = grid.len();
    let n = grid[0].len();
    let mut prev = vec![0; n];

    for i in 0..m {
        let mut curr = vec![0; n];
        for j in 0..n {
            if i == 0 && j == 0 {
                curr[j] = grid[i][j];
            } else {
                let up = if i > 0 {
                    prev[j]
                } else {
                    i32::MAX
                };

                let left = if j > 0 {
                    curr[j - 1]
                } else {
                    i32::MAX
                };

                curr[j] = grid[i][j] + std::cmp::min(up, left);
            }
        }
        prev = curr;
    }

    prev[n - 1]
}

///This problem is of the type where the starting point is fixed and the ending point is variable
///Find the least path sum from the top of the triangle to the bottom (any part of the bottom).
///We can only move to lower adjacent index (from i,j to i + 1, j/j+1) (straight down/right down)
pub fn minimum_triangle(triangle: Vec<Vec<i32>>) -> i32 { //This is the final space optimized code
    let height = triangle.len();
    if height == 1 { //Edge Case
        return triangle[0][0];
    }
    let mut prev = triangle[height - 1].clone(); //Base Case

    for i in (0..=height - 2).rev() {
        let mut curr = vec![0; height];
        for j in (0..=i).rev() {
            let straight_down = triangle[i][j] + prev[j]; //Cost to move straight down in triangle
            let right_down = triangle[i][j] + prev[j + 1]; //Cost to move down+right
            curr[j] = std::cmp::min(straight_down, right_down);
        }
        prev = curr;
    }

    prev[0]
}

///This problem is of the type where both the starting and ending points are variable
///Find the least path sum such that we start from the first row and end at the last row
///We can move from (row, col) -> (row + 1, col - 1)/(row + 1, col)/(row + 1, col + 1)
pub fn min_falling_path_sum(matrix: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (matrix.len(), matrix[0].len());
    let mut dp = vec![vec![-1; cols]; rows];

    dp[0] = matrix[0].clone(); //Base Case

    for i in 1..rows {
        for j in 0..cols {
            let straight_up = dp[i - 1][j];
            let left_up = j.checked_sub(1).map_or(i32::MAX, |col| dp[i - 1][col]);
            let right_up = *dp[i - 1].get(j + 1).unwrap_or(&i32::MAX);

            dp[i][j] = matrix[i][j] + *[straight_up, left_up, right_up].iter().min().unwrap();
        }
    }

    *dp.last().unwrap().iter().min().unwrap()
}
pub fn min_falling_path_sum_optimised(matrix: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (matrix.len(), matrix[0].len());
    let mut prev = matrix[0].clone();

    for i in 1..rows {
        let mut curr = vec![0; cols];
        for (j, _) in matrix[i].iter().enumerate() {
            let straight_up = prev[j];
            let left_up = j.checked_sub(1).map_or(i32::MAX, |col| prev[col]);
            let right_up = *prev.get(j + 1).unwrap_or(&i32::MAX);

            curr[j] = matrix[i][j] + *[straight_up, left_up, right_up].iter().min().unwrap();
        }
        prev = curr;
    }

    *prev.iter().min().unwrap()
}

///This is a problem with fixed starting points and variable ending points.
///We need to maximize the sum of chocolates collected by Alice and Bob.
///They can move straight-down, left-down and right-down. If they are in the same cell, only one of them collects the chocolates
pub fn maximum_chocolates(grid: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (grid.len(), grid[0].len());
    let mut dp = vec![vec![vec![-1; cols]; cols]; rows]; //3-D DP

    //Base Case
    for alice in 0..cols {
        for bob in 0..cols {
            if alice == bob {
                dp[rows - 1][alice][bob] = grid[rows - 1][alice];
            } else {
                dp[rows - 1][alice][bob] = grid[rows - 1][alice] + grid[rows - 1][bob];
            }
        }
    }

    const DELTA_COL: [isize; 3] = [-1, 0, 1];

    for row in (0..=rows - 2).rev() {
        for alice in 0..cols {
            for bob in 0..cols {
                let mut chocolates = 0;
                for &delta_alice in &DELTA_COL {
                    for &delta_bob in &DELTA_COL {
                        let mut val = if alice == bob {
                            grid[row][alice]
                        } else {
                            grid[row][alice] + grid[row][bob]
                        };

                        if alice as isize + delta_alice >= 0
                            && alice as isize + delta_alice < cols as isize
                            && bob as isize + delta_bob >= 0
                            && bob as isize + delta_bob < cols as isize {
                            val += dp[row + 1][(alice as isize + delta_alice) as usize][(bob as isize + delta_bob) as usize];
                        }

                        chocolates = std::cmp::max(chocolates, val);
                    }
                }
                dp[row][alice][bob] = chocolates;
            }
        }
    }

    dp[0][0][cols - 1]
}
pub fn maximum_chocolates_improvement(grid: Vec<Vec<i32>>) -> i32 {
    let (rows, cols) = (grid.len(), grid[0].len());
    let mut dp = vec![vec![vec![0; cols]; cols]; rows];

    for alice in 0..cols {
        for bob in 0..cols {
            dp[rows - 1][alice][bob] = if alice == bob {
                grid[rows - 1][alice]
            } else {
                grid[rows - 1][alice] + grid[rows - 1][bob]
            };
        }
    }

    const DELTA_COL: [isize; 3] = [-1, 0, 1];
    for row in (0..rows - 1).rev() {
        for alice in 0..cols {
            for bob in 0..cols {
                dp[row][alice][bob] = DELTA_COL.iter().cartesian_product(DELTA_COL.iter())
                    .filter_map(|(&delta_alice, &delta_bob)| {
                        let alice_next = (alice as isize + delta_alice).clamp(0, (cols - 1) as isize) as usize;
                        let bob_next = (bob as isize + delta_bob).clamp(0, (cols - 1) as isize) as usize;
                        let val = if alice == bob {
                            grid[row][alice]
                        } else {
                            grid[row][alice] + grid[row][bob]
                        };
                        Some(val + dp[row + 1][alice_next][bob_next])
                    })
                    .max().unwrap()
            }
        }
    }

    dp[0][0][cols - 1]
}

//Now we see some problems on DP on subsequences
pub fn subset_sum_recursive(arr: Vec<i32>, target: i32) -> bool {
    fn helper(index: usize, arr: &Vec<i32>, target: i32) -> bool {
        if target == 0 {
            return true;
        }
        if index == 0 {
            return arr[0] == target;
        }

        let not_take = helper(index - 1, arr, target);
        let take = if target >= arr[index] {
            helper(index - 1, arr, target - arr[index])
        } else {
            false
        };

        take || not_take
    }

    helper(arr.len() - 1, &arr, target)
}

pub fn subset_sum_tabulation(arr: Vec<i32>, target: i32) -> bool {
    let target_usize = target as usize;
    let mut dp = vec![vec![false; target_usize + 1]; arr.len()];

    for i in 0..arr.len() {
        dp[i][0] = true;
    }
    if arr[0] as usize <= target_usize {
        dp[0][arr[0] as usize] = true;
    }

    for i in 1..arr.len() {
        for j in 1..=target_usize {
            let not_take = dp[i - 1][j];
            let take = if arr[i] as usize <= j {
                dp[i - 1][j - arr[i] as usize]
            } else {
                false
            };
            dp[i][j] = not_take || take;
        }
    }

    dp[arr.len() - 1][target as usize]
}

///Use all the elements in the array to find two subsets with an equal sum
pub fn can_partition(nums: Vec<i32>) -> bool {
    //If sum(nums) => odd => false, else, find one subarray with sum = sum(nums)/2
    let sum = nums.iter().sum::<i32>();

    if sum % 2 == 1 {
        return false;
    }

    subset_sum_tabulation(nums, sum / 2)
}

///It is guaranteed that nums has even elements. Find the minimum possible difference between any two subsets
pub fn minimum_difference(nums: Vec<i32>) -> i32 {
    //We check the last row of the dp from subset_sum and find the possible partition sums
    let total_sum: i32 = nums.iter().sum();
    let total_sum = total_sum as usize;
    let mut dp = vec![vec![false; total_sum + 1]; nums.len()];

    for i in 0..nums.len() {
        dp[i][0] = true;
    }
    dp[0][nums[0] as usize] = true;

    for i in 1..nums.len() {
        for j in 1..=total_sum {
            let not_take = dp[i - 1][j];
            let take = if nums[i] as usize <= j {
                dp[i - 1][j - nums[i] as usize]
            } else {
                false
            };
            dp[i][j] = not_take || take;
        }
    }

    let mut min = i32::MAX;
    for i in 0..=total_sum / 2 {
        if dp[nums.len() - 1][i] {
            min = min.min(total_sum.abs_diff(2 * i) as i32);
        }
    }

    min
}

///Find the number of subsets with sum = target (this problem assumes 1<= arr[i] <= 1000)
pub fn perfect_sum(arr: Vec<i32>, target: i32) -> i32 {
    let target_usize = target as usize;
    let mut dp = vec![vec![0; target_usize + 1]; arr.len()];
    //Base Case
    for i in 0..arr.len() {
        dp[i][0] = 1;
    }
    if arr[0] <= target {
        dp[0][arr[0] as usize] = 1;
    }

    for i in 1..arr.len() {
        for j in 0..=target_usize {
            let not_take = dp[i - 1][j];
            let take = if arr[i] as usize <= j {
                dp[i - 1][j - arr[i] as usize]
            } else {
                0
            };
            dp[i][j] = take + not_take;
        }
    }

    dp[arr.len() - 1][target_usize]
}

///The same problem but arr[i] may be 0
pub fn perfect_sum_with_zeroes(arr: Vec<i32>, target: i32) -> i32 {
    //The simple way is to multiply with 2^(number of zeros) (power set)
    let mut dp = vec![vec![-1; target as usize + 1]; arr.len()];
    fn helper(arr: &Vec<i32>, index: usize, sum: i32, dp: &mut Vec<Vec<i32>>) -> i32 {
        /*        if sum == 0 {
                    return 1;
                }
         */
        if index == 0 {
            if sum == 0 && arr[0] == 0 {
                return 2;
            }
            if sum == 0 || sum == arr[0] {
                return 1;
            }
            return 0;
        }

        if dp[index][sum as usize] != -1 {
            return dp[index][sum as usize];
        }

        let not_take = helper(arr, index - 1, sum, dp);
        let take = if arr[index] <= sum {
            helper(arr, index - 1, sum - arr[index], dp)
        } else {
            0
        };

        dp[index][sum as usize] = not_take + take;
        return dp[index][sum as usize];
    }

    helper(&arr, arr.len() - 1, target, &mut dp)
}

///Find the number of subsets such that difference of sums = given difference
pub fn count_partitions(arr: Vec<i32>, difference: i32) -> i32 { //This is also the solution for the problem find_target_sum_ways 494.
    let total_sum: i32 = arr.iter().sum();

    if (total_sum - difference < 0) || (total_sum - difference) % 2 == 1 {
        return 0;
    }

    perfect_sum_with_zeroes(arr, (total_sum - difference) / 2)
}

///Given a bag of weight of W and items with (weight, profit) (items cannot be broken for partial weight). Find the most profit
pub fn knapsack_tabulation(capacity: usize, items: Vec<(usize, usize)>) -> usize { //Items -> (weight, value)
    let mut dp = vec![vec![0; capacity + 1]; items.len()];
    for weight in items[0].0..=capacity { //Base Case
        dp[0][weight] = items[0].1; //If the capacity is more than the weight of first (only) element, take it
    }

    for index in 1..items.len() {
        for weight in 0..=capacity {
            let not_take = dp[index - 1][weight];
            let take = if items[index].0 <= weight {
                items[index].1 + dp[index - 1][weight - items[index].0]
            } else {
                usize::MIN
            };

            dp[index][weight] = std::cmp::max(not_take, take);
        }
    }

    dp[items.len() - 1][capacity]
}

pub fn knapsack_optimised(capacity: usize, items: Vec<(usize, usize)>) -> usize {
    //We can use the 2-D -> 2 1-D array optimization
    //Since only prev is used in the calculation of curr, we can change (0..=capacity) -> (0..=capacity).rev()
    let mut dp = vec![0; capacity + 1];
    //Base Case
    for weight in items[0].0..=capacity {
        dp[weight] = items[0].1;
    }

    for index in 1..items.len() {
        for weight in (0..=capacity).rev() {
            let not_take = dp[weight];
            let take = if items[index].0 <= weight {
                items[index].1 + dp[weight - items[index].0]
            } else {
                usize::MIN
            };

            dp[weight] = std::cmp::max(not_take, take); //Store the result in prev
        }
    }

    dp[capacity]
}

pub fn coin_change(coins: Vec<i32>, amount: i32) -> i32 { //This is the optimal approach
    let amount_usize = amount as usize;
    let mut prev = vec![i32::MAX; amount_usize + 1];

    for sum in 0..=amount_usize {
        if sum as i32 % coins[0] == 0 {
            prev[sum] = sum as i32 / coins[0];
        }
    }

    for &coin in coins.iter().skip(1) { //Skip the first coin as already taken (Taking index does the same thing)
        let mut curr = vec![0; amount_usize + 1];
        for sum in 0..=amount_usize {
            let not_take = prev[sum];
            let take = if coin <= sum as i32 {
                1_i32.saturating_add(curr[sum - coin as usize])
            } else {
                i32::MAX
            };

            curr[sum] = std::cmp::min(not_take, take);
        }

        prev = curr
    }

    if prev[amount_usize] == i32::MAX {
        -1
    } else {
        prev[amount_usize]
    }
}

///This is just an experiment
pub fn coin_change_parallel(coins: Vec<i32>, amount: i32) -> i32 {
    let amount_usize = amount as usize;
    let dp = Mutex::new(vec![i32::MAX; amount_usize + 1]);
    {
        let mut dp_guard = dp.lock().unwrap();
        dp_guard[0] = 0;
    }

    coins.par_iter().for_each(|&coin| {
        let mut local_dp = dp.lock().unwrap();
        for a in coin as usize..=amount_usize {
            if let Some(new_val) = local_dp[a - coin as usize].checked_add(1) {
                local_dp[a] = local_dp[a].min(new_val);
            }
        }
    });

    let dp_guard = dp.lock().unwrap();
    if dp_guard[amount_usize] == i32::MAX { -1 } else { dp_guard[amount_usize] }
}

///Find the number of ways to make amount from the denominations
pub fn coin_change_2(coins: Vec<i32>, amount: i32) -> i32 { //This is the optimal approach using only 1 array
    let amount_usize = amount as usize;
    let mut prev = vec![0; amount_usize + 1];
    //Base Case
    for sum in 0..=amount_usize {
        prev[sum] = if (sum as i32) % coins[0] == 0 {
            1
        } else {
            0
        };
    }

    for &coin in coins.iter().skip(1) {
        for sum in 0..=amount_usize {
            let not_take = prev[sum];
            let take = if coin <= sum as i32 {
                prev[sum - coin as usize]
            } else {
                0
            };

            prev[sum] = not_take + take;
        }
    }

    prev[amount_usize]
}

///prices[i] is the cost of cutting a rod of length i
pub fn cut_rod(prices: Vec<i32>, rod_length: i32) -> i32 { //This is the optimized solution
    let rod_length = rod_length as usize;
    let mut dp = vec![0; rod_length + 1];
    //Base Case
    for length in 0..=rod_length {
        dp[length] = length as i32 * prices[0];
    }

    for index in 1..rod_length {
        for length in 0..=rod_length {
            let not_take = dp[length];
            let take = if index + 1 <= length { //The length of the rod is index + 1
                prices[index] + dp[length - index - 1]
            } else {
                i32::MIN
            };

            dp[length] = std::cmp::max(not_take, take);
        }
    }

    dp[rod_length]
}

pub fn longest_common_subsequence(s1: String, s2: String) -> i32 { //This can be optimized even further to use O(2*n) space
    let (n, m) = (s1.len(), s2.len());
    let (s1, s2) = (s1.chars().collect::<Vec<char>>(), s2.chars().collect::<Vec<char>>());
    //Shifting of index -> index i represents i-1
    let mut dp = vec![vec![0; m + 1]; n + 1];
    //Base Case
    for i in 0..=n {
        dp[i][0] = 0;
    }
    for j in 0..=m {
        dp[0][j] = 0;
    }

    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if s1[i - 1] == s2[j - 1] {
                1 + dp[i - 1][j - 1]
            } else {
                std::cmp::max(dp[i - 1][j], dp[i][j - 1])
            };
        }
    }

    dp[n][m]
}

pub fn longest_common_substring(s1: &str, s2: &str) -> i32 {
    //This follows the same code as the subsequence problem
    let (n, m) = (s1.len(), s2.len());
    let (s1, s2) = (s1.chars().collect::<Vec<char>>(), s2.chars().collect::<Vec<char>>());

    //Shifting of index -> index i represents i-1
    let mut dp = vec![vec![0; m + 1]; n + 1];
    //Base Case
    for i in 0..=n {
        dp[i][0] = 0;
    }
    for j in 0..=m {
        dp[0][j] = 0;
    }

    for i in 1..=n {
        for j in 1..=m {
            if s1[i - 1] == s2[j - 1] {
                dp[i][j] = 1 + dp[i - 1][j - 1]; //Otherwise, the substring resets and dp[i][j]=0 (by default)
            }
        }
    }

    *dp.iter().flatten().max().unwrap()
}

///A supersequence s is such that it contains s1 and s2 as subsequences
pub fn longest_common_supersequence(s1: &str, s2: &str) -> String {
    //Finding the longest common subsequence
    let (n, m) = (s1.len(), s2.len());
    let (s1, s2) = (s1.chars().collect::<Vec<char>>(), s2.chars().collect::<Vec<char>>());
    //Shifting of index -> index i represents i-1
    let mut dp = vec![vec![0; m + 1]; n + 1];
    //Base Case
    for i in 0..=n {
        dp[i][0] = 0;
    }
    for j in 0..=m {
        dp[0][j] = 0;
    }

    for i in 1..=n {
        for j in 1..=m {
            dp[i][j] = if s1[i - 1] == s2[j - 1] {
                1 + dp[i - 1][j - 1]
            } else {
                std::cmp::max(dp[i - 1][j], dp[i][j - 1])
            };
        }
    }

    let mut supersequence = String::new();
    let (mut i, mut j) = (n, m);

    while i > 0 && j > 0 {
        if s1[i - 1] == s2[j - 1] {
            supersequence.push(s1[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            supersequence.push(s1[i - 1]);
            i -= 1;
        } else {
            supersequence.push(s2[j - 1]);
            j -= 1;
        }
    }

    // Add remaining characters from s1 or s2
    while i > 0 {
        supersequence.push(s1[i - 1]);
        i -= 1;
    }
    while j > 0 {
        supersequence.push(s2[j - 1]);
        j -= 1;
    }

    // Reverse the supersequence to get the correct order
    supersequence.chars().rev().collect()
}

pub fn num_distinct(s1: &str, s2: &str) -> i32 {
    let (n, m) = (s1.len(), s2.len());
    let (s1, s2) = (s1.chars().collect::<Vec<char>>(), s2.chars().collect::<Vec<char>>());
    //Shifting the index
    let mut prev = vec![0; m + 1];
    //Base Case
    for _ in 0..=m {
        prev[0] = 1;
    }

    for i in 1..=n {
        let mut curr = vec![0; m + 1];
        curr[0] = 1;
        for j in 1..=m {
            curr[j] = if s1[i - 1] == s2[j - 1] {
                prev[j - 1] + prev[j]
            } else {
                prev[j]
            };
        }
        prev = curr;
    }

    prev[m]
}

pub fn min_distance(word1: &str, word2: &str) -> i32 {
    let (n, m) = (word1.len(), word2.len());
    let (word1, word2) = (word1.chars().collect::<Vec<char>>(), word2.chars().collect::<Vec<char>>());
    //Shifting index
    let mut prev = vec![0; m + 1];
    for j in 0..=m { //Base Case
        prev[j] = j as i32;
    }

    for i in 1..=n {
        let mut curr = vec![0; m + 1];
        curr[0] = i as i32; //Base Case
        for j in 1..=m {
            curr[j] = if word1[i - 1] == word2[j - 1] {
                prev[j - 1]
            } else {
                let insertion = 1 + curr[j - 1];
                let deletion = 1 + prev[j];
                let replace = 1 + prev[j - 1];

                std::cmp::min(insertion, std::cmp::min(deletion, replace))
            };
        }
        prev = curr;
    }

    prev[m]
}

///? -> any single character. * -> any sequence of characters of length >= 0
pub fn is_match(word: &str, pattern: &str) -> bool {
    let (n, m) = (word.len(), pattern.len());
    let (word, pattern) = (word.chars().collect::<Vec<char>>(), pattern.chars().collect::<Vec<char>>());
    //Shifting the index
    let mut prev = vec![false; m + 1];
    prev[0] = true; //Empty word matches with an empty pattern
    //If the word is empty, the pattern matches if it is made of only '*'
    for j in 1..=m {
        prev[j] = prev[j - 1] && pattern[j - 1] == '*';
    }
    for i in 1..=n {
        let mut curr = vec![false; m + 1];
        for j in 1..=m {
            curr[j] = if (word[i - 1] == pattern[j - 1]) || pattern[j - 1] == '?' { //If the characters match
                prev[j - 1]
            } else if pattern[j - 1] == '*' {
                prev[j] | curr[j - 1] //Take consecutive characters from input | take 0 characters
            } else {
                false
            };
        }
        prev = curr;
    }

    prev[m]
}

//Best Time to Buy and Sell Stock already done in arrays

///In this problem, we can buy and sell multiple times, but we cannot have more than one stock at a time
pub fn max_profit_2_tabulation(prices: Vec<i32>) -> i32 {
    let mut prev = vec![0; 2];

    for day in (0..prices.len()).rev() {
        let mut curr = vec![0; 2];
        for can_buy in 0..=1 {
            if can_buy == 1 {
                curr[can_buy] = std::cmp::max(-prices[day] + prev[0], prev[1]);
            } else {
                curr[can_buy] = std::cmp::max(prices[day] + prev[1], prev[0]);
            }
        }
        prev = curr;
    }

    prev[1]
}

pub fn max_profit_2(prices: Vec<i32>) -> i32 {
    //Since the vector is only of 2 variables, we can reduce the space complexity
    let (mut ahead_buy, mut ahead_not_buy) = (0, 0);

    for day in (0..prices.len()).rev() {
        let curr_buy = std::cmp::max(-prices[day] + ahead_not_buy, ahead_buy);
        let curr_not_buy = std::cmp::max(prices[day] + ahead_buy, ahead_not_buy);

        (ahead_buy, ahead_not_buy) = (curr_buy, curr_not_buy);
    }

    ahead_buy
}

//Buy and Sell Stock 3 is a special case of Buy and Sell Stock 4 where max_transactions = 2
pub fn max_profit_4(prices: Vec<i32>, max_transactions: usize) -> i32 {
    let mut prev = vec![vec![0; max_transactions + 1]; 2];

    for &price in prices.iter().rev() {
        let mut curr = vec![vec![0; max_transactions + 1]; 2];
        for can_buy in 0..=1 {
            for transaction in 1..=max_transactions {
                curr[can_buy][transaction] = if can_buy == 1 {
                    let buy = -price + prev[0][transaction];
                    let not_buy = prev[1][transaction];
                    std::cmp::max(buy, not_buy)
                } else {
                    let sell = price + prev[1][transaction - 1];
                    let not_sell = prev[0][transaction];
                    std::cmp::max(sell, not_sell)
                };
            }
        }
        prev = curr;
    }

    prev[1][max_transactions]
}

pub fn max_profit_with_cooldown(prices: Vec<i32>) -> i32 { //This can't be further space optimized since we require index + 1 and index + 2
    let mut dp = vec![vec![0; 2]; prices.len() + 2];

    for (index, &price) in prices.iter().enumerate().rev() {
        dp[index][1] = std::cmp::max(
            -price + dp[index + 1][0],
            dp[index + 1][1],
        );
        dp[index][0] = std::cmp::max(
            price + dp[index + 2][1], //dp has index + 2 to take care of this
            dp[index + 1][0],
        );
    }

    dp[0][1]
}

///T.C. = O(N^2). S.C. = O(N^2) + O(N) (Stack)
pub fn longest_increasing_sequence_memoization(arr: Vec<i32>) -> i32 {
    fn helper(index: usize, prev_index: isize, arr: &Vec<i32>, dp: &mut Vec<Vec<i32>>) -> i32 {
        if index == arr.len() {
            return 0;
        }
        if dp[index][(prev_index + 1) as usize] != -1 {
            return dp[index][(prev_index + 1) as usize];
        }

        let take = if prev_index == -1 || arr[index] > arr[prev_index as usize] {
            1 + helper(index + 1, index as isize, arr, dp)
        } else {
            0
        };

        let not_take = helper(index + 1, prev_index, arr, dp);

        dp[index][(prev_index + 1) as usize] = std::cmp::max(take, not_take);

        dp[index][(prev_index + 1) as usize]
    }
    let mut dp = vec![vec![-1; arr.len() + 1]; arr.len() + 1];
    helper(0, -1, &arr, &mut dp)
}

///T.C. ≅ O(N^2). S.C. = O(N)
pub fn longest_increasing_subsequence_tabulation(arr: Vec<i32>) -> (usize, Vec<i32>) {
    if arr.is_empty() {
        return (0, vec![]);
    }

    let mut dp: Vec<usize> = vec![1; arr.len()]; //dp[i] -> LIS that ends at index 'i'
    let mut hash: Vec<usize> = (0..arr.len()).collect(); //Predecessors in LIS

    for index in 0..arr.len() {
        for prev_index in 0..index {
            if arr[prev_index] < arr[index] && 1 + dp[prev_index] > dp[index] {
                dp[index] = std::cmp::max(dp[index], 1 + dp[prev_index]); //Extending the LIS
                hash[index] = prev_index;
            }
        }
    }

    let (mut max_index, &max_len) = dp.iter().enumerate().max_by_key(|&(_, &val)| val).unwrap();

    let mut lis = Vec::new();
    for _ in 0..max_len {
        lis.push(arr[max_index]);
        max_index = hash[max_index];
    }
    lis.reverse();

    (max_len, lis)
}

///T.C. = O(N*log n). S.C. = O(N)
pub fn longest_increasing_subsequence_binary_search(arr: Vec<i32>) -> usize {
    let mut tails = vec![];

    for &x in arr.iter() {
        match tails.binary_search(&x) {
            Ok(_) => {} //If x is already present, do nothing
            Err(i) => {
                if i == tails.len() {
                    tails.push(x); //If x is greater than all elements, push it
                } else {
                    tails[i] = x; //Replace the fist element greater than x with x
                }
            }
        }
    }

    tails.len() //Tails will be a valid LIS
}

///Largest subset where forall i, j, s[i] % s[j] == 0 or s[j] % s[i] == 0
pub fn largest_divisible_subset(mut arr: Vec<i32>) -> Vec<i32> {
    arr.sort(); //Sort so that the larger element comes after
    let size = arr.len();
    let mut dp = vec![1; size];
    let mut hash: Vec<usize> = (0..size).collect();

    for i in 0..size {
        for j in 0..i {
            if arr[i] % arr[j] == 0 && dp[i] < dp[j] + 1 {
                dp[i] = std::cmp::max(dp[i], 1 + dp[j]);
                hash[i] = j;
            }
        }
    }

    let (mut max_index, &length) = dp.iter().enumerate().max_by_key(|&(_, &val)| val).unwrap();
    let mut sequence = Vec::new();
    for _ in 0..length {
        sequence.push(arr[max_index]);
        max_index = hash[max_index];
    }

    sequence
}

pub fn longest_str_chain(mut words: Vec<String>) -> i32 {
    words.sort_unstable_by(|a, b| a.len().cmp(&b.len()));
    fn difference_of_one_char(smaller: Vec<char>, larger: Vec<char>) -> bool {
        if smaller.len() + 1 != larger.len() {
            return false;
        }

        let mut skipped = false;
        let (mut i, mut j) = (0, 0);
        while i < smaller.len() {
            if smaller[i] != larger[j] {
                if skipped {
                    return false;
                }
                skipped = true;
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }

        true
    }
    let mut dp = vec![1; words.len()];

    for i in 0..words.len() {
        for j in 0..i {
            if difference_of_one_char(words[j].chars().collect(), words[i].chars().collect()) && dp[i] < 1 + dp[j] {
                dp[i] = 1 + dp[j];
            }
        }
    }

    *dp.iter().max().unwrap()
}

///A chain is made when the next character has one extra character at any position
pub fn longest_str_chain_best(mut words: Vec<String>) -> i32 {
    words.sort_unstable_by(|a, b| a.len().cmp(&b.len()));
    let mut dp = HashMap::new(); //Word: LIS ending with word
    let mut longest_chain = 0;

    for word in words.iter() {
        let mut best_chain_length = 1; //Single word is a chain of length 1

        for i in 0..word.len() { //Form all possible predecessors for current word
            let mut predecessor = word.clone();
            predecessor.remove(i); //Remove the char

            if let Some(&prev_chain_length) = dp.get(&predecessor) {
                best_chain_length = best_chain_length.max(prev_chain_length + 1);
            }
        }

        dp.insert(word.clone(), best_chain_length);
        longest_chain = longest_chain.max(best_chain_length);
    }

    longest_chain
}

///A bitnoic sequence is one that is first strictly increasing and then strictly decreasing
pub fn longest_bitonic_sequences(arr: Vec<i32>) -> i32 {
    //Find the LIS from the left end, then the right end
    let mut dp_front = vec![1; arr.len()];
    for i in 0..arr.len() {
        for j in 0..i {
            if arr[j] < arr[i] && dp_front[i] < 1 + dp_front[j] {
                dp_front[i] = 1 + dp_front[j];
            }
        }
    }
    let mut dp_back = vec![1; arr.len()];
    for i in (0..arr.len()).rev() {
        for j in (i..arr.len()).rev() {
            if arr[j] < arr[i] && dp_back[i] < dp_back[j] + 1 {
                dp_back[i] = 1 + dp_back[j];
            }
        }
    }

    let mut max_len = 0;
    for i in 0..arr.len() {
        if dp_front[i] != -1 && dp_back[i] != -1 { //Remove all monotonic sequences
            max_len = max_len.max(dp_front[i] + dp_back[i] - 1); // -1 to remove the element itself from being counted twice
        }
    }

    max_len
}

pub fn number_of_lis(arr: Vec<i32>) -> i32 {
    let mut dp = vec![1; arr.len()];
    let mut count = vec![1; arr.len()];
    for i in 0..arr.len() {
        for j in 0..i {
            if arr[j] < arr[i] && dp[i] < 1 + dp[j] { //If the subsequence increases
                dp[i] = 1 + dp[j];
                count[i] = count[j]; //The number of ways to reach this subsequence is the same
            } else if arr[j] < arr[i] && dp[i] == 1 + dp[j] { //If another number can form the same length subsequence
                count[i] += count[j]; //The number of ways to reach increases
            }
        }
    }

    let max_len = *dp.iter().max().unwrap();
    let mut ways = 0;
    for i in 0..arr.len() {
        if dp[i] == max_len {
            ways += count[i];
        }
    }

    ways
}

///The number of matrices = matrices.size() - 1, two adjacent entries make up the dimensions of the matrix
pub fn matrix_multiplication_memoization(matrices: Vec<i32>) -> i32 {
    //The dimensions of the ith matrix is matrices[i-1] ✖️ matrices[i]
    //The number of operations to multiply two matrices (m*n) ✖️ (n*p) = m*n*p
    ///i..j is the minimum operations to multiply from matrix i to matrix j
    fn helper(i: usize, j: usize, matrices: &Vec<i32>, dp: &mut Vec<Vec<i32>>) -> i32 {
        if i == j { //Base Case
            return 0; //No operations for single matrix
        }
        if dp[i][j] != -1 {
            return dp[i][j];
        }

        let mut minimum_operations = i32::MAX;
        for k in i..j {
            let operations = matrices[i - 1] * matrices[k] * matrices[j] + helper(i, k, matrices, dp) + helper(k + 1, j, matrices, dp);
            minimum_operations = minimum_operations.min(operations);
        }

        minimum_operations
    }
    let mut dp = vec![vec![-1; matrices.len()]; matrices.len()];
    //Instead of n^2 we can use (n-1)^2 by shifting the index so that matrix[i] represents i + 1 (This is because i != 0 so we can save that space but in theory S.C. = O(n^2))
    helper(1, matrices.len() - 1, &matrices, &mut dp)
}

pub fn matrix_multiplication(matrices: Vec<i32>) -> i32 { //This can't be further optimized
    let mut dp = vec![vec![0; matrices.len()]; matrices.len()];
    //Base Case -> i == j => dp[i][j] = 0
    for i in (1..matrices.len()).rev() {
        for j in i + 1..matrices.len() { //Starting j from i + 1 since the end of the partition must be after the beginning
            let mut minimum_operations = i32::MAX;
            for k in i..j {
                let operations = matrices[i - 1] * matrices[k] * matrices[j] + dp[i][k] + dp[k + 1][j];
                minimum_operations = minimum_operations.min(operations);
            }
            dp[i][j] = minimum_operations;
        }
    }

    dp[1][matrices.len() - 1]
}

///UNTESTED
pub fn matrix_chain_multiplication(matrices: Vec<i32>) -> (i32, Vec<Vec<usize>>) {
    let mut dp = vec![vec![i32::MAX; matrices.len()]; matrices.len()];
    let mut split = vec![vec![0; matrices.len()]; matrices.len()];
    for i in 1..matrices.len() { //Base Case
        dp[i][i] = 0;
    }
    for i in (1..matrices.len() - 1).rev() {
        for j in i + 1..matrices.len() {
            for k in i..j {
                let cost = dp[i][k] + dp[k + 1][j] + matrices[i - 1] * matrices[k] * matrices[j];
                if cost < dp[i][j] {
                    dp[i][j] = cost;
                    split[i][j] = k; //Find the index to partition the matrices
                }
            }
        }
    }

    (dp[1][matrices.len() - 1], split)
}

pub fn optimal_parenthesis(split: &Vec<Vec<usize>>, i: usize, j: usize) -> String {
    let subscript_chars = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];

    let to_subscript = |index: usize| -> String {
        index.to_string().chars().map(|c| {
            subscript_chars[c.to_digit(10).unwrap() as usize]
        }).collect()
    };

    if i == j {
        format!("A{}", to_subscript(i))
    } else {
        format!(" ( {}{}) ",
                optimal_parenthesis(split, i, split[i][j]),
                optimal_parenthesis(split, split[i][j] + 1, j),
        )
    }
}

///Cuts[i] is the places on the stick where we are allowed to cut. The cost of cutting a stick is the length of the stick (before the cut)
pub fn min_cost(length: i32, cuts: Vec<i32>) -> i32 {
    let mut cuts = {
        let mut c = Vec::with_capacity(cuts.len() + 2);
        c.push(0);
        c.extend(cuts);
        c.push(length);
        c
    };
    cuts.sort_unstable();
    let mut dp = vec![vec![0; cuts.len()]; cuts.len()];

    for i in (1..cuts.len() - 1).rev() {
        for j in i..cuts.len() - 1 {
            dp[i][j] = (i..=j)
                .map(|k| cuts[j + 1] - cuts[i - 1] + dp[i][k - 1] + dp[k + 1][j])
                .min().unwrap_or(0);
        }
    }

    dp[1][cuts.len() - 2]
}

pub fn max_coins(nums: Vec<i32>) -> i32 {
    let nums = {
        let mut n = Vec::with_capacity(nums.len() + 2);
        n.push(1);
        n.extend(nums);
        n.push(1);
        n
    };
    let mut dp = vec![vec![0; nums.len()]; nums.len()];

    for i in (1..nums.len() - 1).rev() {
        for j in i..nums.len() - 1 {
            dp[i][j] = (i..=j)
                .map(|k| nums[i - 1] * nums[k] * nums[j + 1] + dp[i][k - 1] + dp[k + 1][j])
                .max().unwrap_or(0);
        }
    }

    dp[1][nums.len() - 2]
}

///Find the number of ways to parse a boolean expression so that the result is true
pub fn number_of_ways_to_true(expression: String) -> i32 {
    let expression: Vec<char> = expression.chars().collect();
    let mut dp = HashMap::new();
    fn helper(i: usize, j: usize, is_true: bool, expression: &Vec<char>, dp: &mut HashMap<(usize, usize, bool), i32>) -> i32 {
        if i > j {
            return 0;
        }
        if i == j {
            //If we need a true, and we have a true return 1 else 0
            return if is_true == (expression[i] == 'T') { 1 } else { 0 };
        }

        if let Some(&cached) = dp.get(&(i, j, is_true)) {
            return cached;
        }

        let mut ways = 0;
        for k in (i + 1)..j {
            if ['&', '|', '^'].contains(&expression[k]) { //Only if the partition point is an operator
                let left_true = helper(i, k - 1, true, expression, dp);
                let left_false = helper(i, k - 1, false, expression, dp);
                let right_true = helper(k + 1, j, true, expression, dp);
                let right_false = helper(k + 1, j, false, expression, dp);

                ways += match expression[k] {
                    '&' => { //& is true only if T & T, and false otherwise
                        if is_true {
                            left_true * right_true
                        } else {
                            left_true * right_false + left_false * right_true + left_false * right_false
                        }
                    }
                    '|' => { //| is true if T | T, T | F, F | T and false otherwise
                        if is_true {
                            left_true * right_true + left_true * right_false + left_false * right_true
                        } else {
                            left_false * right_false
                        }
                    }
                    '^' => { //^ is true if T | F, F | T and false otherwise
                        if is_true {
                            left_true * right_false + left_false * right_true
                        } else {
                            left_true * right_true + left_false * right_false
                        }
                    }
                    _ => 0,
                }
            }
        }
        dp.insert((i, j, is_true), ways);
        ways
    }

    helper(0, expression.len() - 1, true, &expression, &mut dp)
}

pub fn palindrome_partitioning(str: String) -> i32 {
    // fn is_palindrome(mut i: usize, mut j: usize, str: &Vec<char>) -> bool {
    //     while i < j {
    //         if str[i] != str[j] {
    //             return false;
    //         }
    //         i += 1;
    //         j -= 1;
    //     }
    //
    //     true
    // }
    let str: Vec<char> = str.chars().collect();
    let mut dp = vec![0; str.len() + 1];
    let mut is_palindrome = vec![vec![false; str.len()]; str.len()];
    //Precompute Palindromes
    for len in 1..=str.len() {
        for start in 0..=str.len() - len {
            let end = start + len - 1;
            is_palindrome[start][end] = str[start] == str[end] && (len <= 2 || is_palindrome[start + 1][end - 1]);
        }
    }
    //Base Case dp[str.len()] = 0;
    for i in (0..str.len()).rev() {
        dp[i] = (i..str.len())
            .map(|j| if is_palindrome[i][j] { 1 + dp[j + 1] } else { i32::MAX })
            .min().unwrap();
    }


    dp[0] - 1
}

pub fn max_sum_after_partitioning(arr: Vec<i32>, max_length: usize) -> i32 {
    let mut dp = vec![0; arr.len() + 1];
    //Base Case- dp[arr.len()] = 0
    for i in (0..arr.len()).rev() {
        dp[i] = (i..arr.len().min(i + max_length))
            .fold((i32::MIN, i32::MIN), |(max_sum, max_value), j| {
                let max_value = max_value.max(arr[j]);
                let sum = (j - i + 1) as i32 * max_value + dp[j + 1];
                (max_sum.max(sum), max_value)
            }).0;
    }

    dp[0]
}

pub fn maximal_rectangle(matrix: Vec<Vec<char>>) -> i32 {
    //Loop over each row finding the maximum area of histograms (in stack_queues.rs) to find the maximum
    let mut heights = vec![0; matrix[0].len()];
    let mut maximum_area = 0;
    for i in 0..matrix.len() {
        for j in 0..matrix[0].len() {
            if matrix[i][j] == '1' {
                heights[j] += 1;
            } else {
                heights[j] = 0;
            }
        }
        maximum_area = maximum_area.max(largest_rectangle_area_optimised(heights.clone()));
    }

    maximum_area
}

pub fn count_squares(matrix: Vec<Vec<i32>>) -> i32 {
    let mut dp = vec![vec![0; matrix[0].len()]; matrix.len()];
    dp[0] = matrix[0].clone(); //Copy first row
    for i in 1..matrix.len() { //Copy first column
        dp[i][0] = matrix[i][0];
    }

    for i in 1..matrix.len() {
        for j in 1..matrix[0].len() {
            if matrix[i][j] == 0 {
                dp[i][j] = 0;
            } else {
                dp[i][j] = 1 + *[dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1]].iter().min().unwrap();
            }
        }
    }

    dp.iter().flatten().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fibonacci_test() {
        assert_eq!(fibonacci(10), 89);
        assert_eq!(fibonacci_better(10), 89);
        assert_eq!(fibonacci_best(10), 89);
    }

    #[test]
    fn climbing_stairs_test() {
        assert_eq!(climb_stairs(2), 2);
        assert_eq!(climb_stairs(3), 3);
    }

    #[test]
    fn frog_jump_test() {
        assert_eq!(frog_jump_recursive(&[10, 20, 30, 10]), 20);
        assert_eq!(frog_jump(&[10, 20, 30, 10]), 20);
        assert_eq!(frog_jump_optimised(&[10, 20, 30, 10]), 20);
        assert_eq!(frog_jump_space_optimised(&[10, 20, 30, 10]), 20);
    }

    #[test]
    fn rob_test() {
        assert_eq!(rob(&[2, 3, 2]), 4);
        assert_eq!(rob(&[1, 2, 3, 1]), 4);
        assert_eq!(rob_optimised(&[2, 3, 2]), 4);
        assert_eq!(rob_optimised(&[1, 2, 3, 1]), 4);
    }

    #[test]
    fn rob_again_test() {
        assert_eq!(rob_2(&[2, 3, 2]), 3);
    }

    #[test]
    fn max_points_test() {
        assert_eq!(maximum_points(vec![
            vec![1, 2, 5],
            vec![3, 1, 1],
            vec![3, 3, 3],
        ]), 11);

        assert_eq!(maximum_points_optimised(vec![
            vec![1, 2, 5],
            vec![3, 1, 1],
            vec![3, 3, 3],
        ]), 11);
    }

    #[test]
    fn unique_paths_test() {
        assert_eq!(unique_paths_recursive(3, 7), 28);
        assert_eq!(unique_paths_recursive(3, 2), 3);
        assert_eq!(unique_paths_dp(3, 7), 28);
        assert_eq!(unique_paths_dp(3, 2), 3);
        assert_eq!(unique_paths(3, 7), 28);
        assert_eq!(unique_paths(3, 2), 3);
    }

    #[test]
    fn unique_paths_with_obstacles_test() {
        assert_eq!(unique_path_with_obstacles(vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ]), 2);
        assert_eq!(unique_path_with_obstacles(vec![
            vec![0, 1],
            vec![0, 0],
        ]), 1);
        assert_eq!(unique_path_with_obstacles_optimised(vec![
            vec![0, 0, 0],
            vec![0, 1, 0],
            vec![0, 0, 0],
        ]), 2);
        assert_eq!(unique_path_with_obstacles_optimised(vec![
            vec![0, 1],
            vec![0, 0],
        ]), 1);
        assert_eq!(unique_path_with_obstacles_optimised(vec![vec![0, 0]]), 1);
    }

    #[test]
    fn min_path_sum_test() {
        assert_eq!(min_path_sum_recursive(vec![
            vec![1, 3, 1],
            vec![1, 5, 1],
            vec![4, 2, 1],
        ]), 7);
        assert_eq!(min_path_sum_recursive(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]), 12);
        assert_eq!(min_path_sum(vec![
            vec![1, 3, 1],
            vec![1, 5, 1],
            vec![4, 2, 1],
        ]), 7);
        assert_eq!(min_path_sum(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]), 12);
        assert_eq!(min_path_sum_optimised(vec![
            vec![1, 3, 1],
            vec![1, 5, 1],
            vec![4, 2, 1],
        ]), 7);
        assert_eq!(min_path_sum_optimised(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]), 12);
    }

    #[test]
    fn minimum_total_test() {
        assert_eq!(minimum_triangle(vec![
            vec![2],
            vec![3, 4],
            vec![6, 5, 7],
            vec![4, 1, 8, 3],
        ]), 11);
        assert_eq!(minimum_triangle(vec![vec![-10]]), -10);
    }

    #[test]
    fn min_falling_path_sum_test() {
        assert_eq!(min_falling_path_sum(vec![
            vec![2, 1, 3],
            vec![6, 5, 4],
            vec![7, 8, 9],
        ]), 13);
        assert_eq!(min_falling_path_sum(vec![
            vec![-19, 57],
            vec![-40, -5],
        ]), -59);
        assert_eq!(min_falling_path_sum_optimised(vec![
            vec![2, 1, 3],
            vec![6, 5, 4],
            vec![7, 8, 9],
        ]), 13);
        assert_eq!(min_falling_path_sum_optimised(vec![
            vec![-19, 57],
            vec![-40, -5],
        ]), -59);
    }

    #[test]
    fn maximum_chocolates_test() {
        assert_eq!(maximum_chocolates(vec![
            vec![3, 1, 1],
            vec![2, 5, 1],
            vec![1, 5, 5],
            vec![2, 1, 1],
        ]), 24);
        assert_eq!(maximum_chocolates_improvement(vec![
            vec![3, 1, 1],
            vec![2, 5, 1],
            vec![1, 5, 5],
            vec![2, 1, 1],
        ]), 24);
    }

    #[test]
    fn subset_sum_test() {
        let arr = vec![3, 34, 4, 12, 5, 2];
        let target1 = 9;
        let target2 = 30;

        assert_eq!(subset_sum_recursive(arr.clone(), target1), true);
        assert_eq!(subset_sum_recursive(arr.clone(), target2), false);

        assert_eq!(subset_sum_tabulation(arr.clone(), target1), true);
        assert_eq!(subset_sum_tabulation(arr.clone(), target2), false);
    }

    #[test]
    fn can_partition_test() {
        assert_eq!(can_partition(vec![1, 5, 11, 5]), true);
        assert_eq!(can_partition(vec![1, 2, 3, 5]), false);
    }

    #[test]
    fn minimum_difference_test() {
        assert_eq!(minimum_difference(vec![3, 9, 7, 3]), 2);
        //assert_eq!(minimum_difference(vec![-36, 36]), 72); This test case does not work as the algo works if nums[i] > 0
        assert_eq!(minimum_difference(vec![1, 2, 3, 4]), 0);
        assert_eq!(minimum_difference(vec![8, 6, 5]), 3);
    }

    #[test]
    fn perfect_sum_test() {
        assert_eq!(perfect_sum(vec![5, 2, 3, 10, 6, 8], 10), 3);
        assert_eq!(perfect_sum(vec![2, 5, 1, 4, 3], 10), 3);
        //assert_eq!(perfect_sum(vec![0, 0, 1], 1), 1);
        assert_eq!(perfect_sum_with_zeroes(vec![0, 0, 1], 1), 4);
    }

    #[test]
    fn count_partitions_test() {
        assert_eq!(count_partitions(vec![1, 1, 1, 1, 1], 3), 5);
    }

    #[test]
    fn knapsack_test() {
        assert_eq!(knapsack_tabulation(4, vec![(4, 1), (5, 2), (1, 3)]), 3);
        assert_eq!(knapsack_tabulation(4, vec![(4, 1), (5, 2), (1, 3)]), 3);

        assert_eq!(knapsack_optimised(3, vec![(4, 1), (5, 2), (6, 3)]), 0);
        assert_eq!(knapsack_optimised(3, vec![(4, 1), (5, 2), (6, 3)]), 0);
    }

    #[test]
    fn coin_change_test() {
        assert_eq!(coin_change(vec![1, 2, 5], 11), 3);
        assert_eq!(coin_change(vec![1, 2, 5], 11), 3);
        assert_eq!(coin_change(vec![2], 3), -1);
        assert_eq!(coin_change(vec![1], 0), 0);

        assert_eq!(coin_change_parallel(vec![2, 5, 10, 1], 27), 4);
        assert_eq!(coin_change_parallel(vec![2], 3), -1);
        assert_eq!(coin_change_parallel(vec![1], 0), 0);
        assert_eq!(coin_change_parallel(vec![2, 5, 10, 1], 27), 4);
    }

    #[test]
    fn coin_change_2_test() {
        assert_eq!(coin_change_2(vec![1, 2, 5], 5), 4);
        assert_eq!(coin_change_2(vec![2], 3), 0);
        assert_eq!(coin_change_2(vec![10], 10), 1);
    }

    #[test]
    fn rod_cutting_test() {
        assert_eq!(cut_rod(vec![1, 5, 8, 9, 10, 17, 17, 20], 8), 22);
        assert_eq!(cut_rod(vec![3, 5, 8, 9, 10, 17, 17, 20], 8), 24);
    }

    #[test]
    fn longest_common_subsequence_test() {
        assert_eq!(longest_common_subsequence("abcde".to_string(), "ace".to_string()), 3);
        assert_eq!(longest_common_subsequence("abc".to_string(), "abc".to_string()), 3);
        assert_eq!(longest_common_subsequence("abc".to_string(), "def".to_string()), 0);
    }

    #[test]
    fn longest_common_substring_test() {
        assert_eq!(longest_common_substring("ABCDGH", "ACDGHR"), 4);
        assert_eq!(longest_common_substring("ABC", "ACB"), 1);
    }

    #[test]
    fn shortest_supersequence_test() {
        assert_eq!(longest_common_supersequence("abac", "cab"), "cabac".to_string());
    }

    #[test]
    fn num_distinct_test() {
        assert_eq!(num_distinct("rabbbit", "rabbit"), 3);
        assert_eq!(num_distinct("babgbag", "bag"), 5);
    }

    #[test]
    fn min_distance_test() {
        assert_eq!(min_distance("horse", "ros"), 3);
        assert_eq!(min_distance("intention", "execution"), 5);
    }

    #[test]
    fn is_match_test() {
        assert_eq!(is_match("aa", "a"), false);
        assert_eq!(is_match("aa", "*"), true);
        assert_eq!(is_match("cb", "?a"), false);
    }

    #[test]
    fn buy_sell_stock_2() {
        assert_eq!(max_profit_2_tabulation(vec![7, 1, 5, 3, 6, 4]), 7);
        assert_eq!(max_profit_2_tabulation(vec![7, 1, 5, 3, 6, 4]), 7);
        assert_eq!(max_profit_2_tabulation(vec![1, 2, 3, 4, 5]), 4);

        assert_eq!(max_profit_2(vec![7, 6, 4, 3, 1]), 0);
        assert_eq!(max_profit_2(vec![1, 2, 3, 4, 5]), 4);
        assert_eq!(max_profit_2(vec![7, 6, 4, 3, 1]), 0);
    }

    #[test]
    fn buy_sell_stock_4() {
        assert_eq!(max_profit_4(vec![2, 4, 1], 2), 2);
        assert_eq!(max_profit_4(vec![3, 2, 6, 5, 0, 3], 2), 7);
    }

    #[test]
    fn buy_sell_stock_cooldown() {
        assert_eq!(max_profit_with_cooldown(vec![1, 2, 3, 0, 2]), 3);
        assert_eq!(max_profit_with_cooldown(vec![1]), 0);
    }

    #[test]
    fn longest_increasing_subsequence() {
        let arr = vec![10, 9, 2, 5, 3, 7, 101, 18];
        assert_eq!(longest_increasing_sequence_memoization(arr.clone()), 4);
        assert_eq!(longest_increasing_subsequence_tabulation(arr.clone()), (4, vec![2, 5, 7, 18]));
        assert_eq!(longest_increasing_subsequence_tabulation(vec![0, 1, 0, 3, 2, 3]), (4, vec![0, 1, 2, 3]));
        assert_eq!(longest_increasing_subsequence_binary_search(arr.clone()), 4);
        assert_eq!(longest_increasing_subsequence_binary_search(vec![0, 1, 0, 3, 2, 3]), 4);
    }

    #[test]
    fn largest_divisible_subset_test() {
        assert_eq!(largest_divisible_subset(vec![1, 2, 3]), vec![3, 1]);
        assert_eq!(largest_divisible_subset(vec![1, 2, 4, 8]), vec![1, 2, 4, 8].into_iter().rev().collect::<Vec<i32>>());
    }

    #[test]
    fn longest_str_chain_test() {
        let words1 = ["a", "b", "ba", "bca", "bda", "bdca"].map(|x| x.to_string()).to_vec();
        let words2 = ["xbc", "pcxbcf", "xb", "cxbc", "pcxbc"].map(|x| x.to_string()).to_vec();
        let words3 = ["abcd", "dbqca"].map(|x| x.to_string()).to_vec();
        assert_eq!(longest_str_chain(words1.clone()), 4);
        assert_eq!(longest_str_chain(words2.clone()), 5);
        assert_eq!(longest_str_chain(words3.clone()), 1);


        assert_eq!(longest_str_chain(words1), 4);
        assert_eq!(longest_str_chain(words2), 5);
        assert_eq!(longest_str_chain(words3), 1);
    }

    #[test]
    fn longest_bitonic_sequence_test() {
        assert_eq!(longest_bitonic_sequences(vec![1, 2, 5, 3, 2]), 5);
        assert_eq!(longest_bitonic_sequences(vec![1, 11, 2, 10, 4, 5, 2, 1]), 6);
    }

    #[test]
    fn number_of_lis_test() {
        assert_eq!(number_of_lis(vec![1, 3, 5, 4, 7]), 2);
        assert_eq!(number_of_lis(vec![2, 2, 2, 2, 2]), 5);
    }

    #[test]
    fn matrix_multiplication_test() {
        assert_eq!(matrix_multiplication_memoization(vec![40, 20, 30, 10, 30]), 26000);
        assert_eq!(matrix_multiplication_memoization(vec![40, 20, 30, 10, 30]), 26000);

        assert_eq!(matrix_multiplication(vec![10, 30, 5, 60]), 4500);
        assert_eq!(matrix_multiplication(vec![10, 30, 5, 60]), 4500);
    }

    #[test]
    fn min_cost_test() {
        assert_eq!(min_cost(7, vec![1, 3, 4, 5]), 16);
        assert_eq!(min_cost(9, vec![5, 6, 1, 4, 2]), 22);
    }

    #[test]
    fn max_coins_test() {
        assert_eq!(max_coins(vec![3, 1, 5, 8]), 167);
        assert_eq!(max_coins(vec![1, 5]), 10);
    }

    #[test]
    fn number_of_ways_to_true_test() {
        assert_eq!(number_of_ways_to_true(String::from("T^T^F")), 0);
        assert_eq!(number_of_ways_to_true(String::from("F|T^F")), 2);
    }

    #[test]
    fn palindrome_partitioning_test() {
        assert_eq!(palindrome_partitioning(String::from("aab")), 1);
        assert_eq!(palindrome_partitioning(String::from("a")), 0);
        assert_eq!(palindrome_partitioning(String::from("ab")), 1);
    }

    #[test]
    fn max_sum_after_partitioning_test() {
        assert_eq!(max_sum_after_partitioning(vec![1, 15, 7, 9, 2, 5, 10], 3), 84);
        assert_eq!(max_sum_after_partitioning(vec![1, 4, 1, 5, 7, 3, 6, 1, 9, 9, 3], 4), 83);
        assert_eq!(max_sum_after_partitioning(vec![1], 1), 1);
    }

    #[test]
    fn maximal_rectangle_test() {
        assert_eq!(maximal_rectangle(vec![
            vec!['1', '0', '1', '0', '0'],
            vec!['1', '0', '1', '1', '1'],
            vec!['1', '1', '1', '1', '1'],
            vec!['1', '0', '0', '1', '0'],
        ]), 6);
        assert_eq!(maximal_rectangle(vec![vec!['0']]), 0);
        assert_eq!(maximal_rectangle(vec![vec!['1']]), 1);
    }

    #[test]
    fn count_squares_test() {
        assert_eq!(count_squares(vec![
            vec![0, 1, 1, 1],
            vec![1, 1, 1, 1],
            vec![0, 1, 1, 1],
        ]), 15);
        assert_eq!(count_squares(vec![
            vec![1, 0, 1],
            vec![1, 1, 0],
            vec![1, 1, 0],
        ]), 7);
    }
}