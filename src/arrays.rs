use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

pub fn second_largest(arr: &[i32]) -> Option<i32> {
    if arr.len() <= 1 {
        return None;
    }

    let mut largest = None;
    let mut second = None;

    for &i in arr {
        match largest {
            Some(l) if i > l => {
                second = largest;
                largest = Some(i);
            }
            Some(l) if i < l => {
                second = match second {
                    Some(s) if i > s => Some(i),
                    None => Some(i),
                    _ => second,
                };
            }
            None => largest = Some(i),
            _ => {}
        }
    }
    second
}

pub fn two_sum(nums: Vec<i32>, target: i32) -> Option<(usize, usize)> {
    let mut map: HashMap<i32, usize> = HashMap::new(); //Element, index

    for (ind, &ele) in nums.iter().enumerate() {
        let more = target - ele;
        if let Some(&index) = map.get(&more) {
            return Some((index, ind));
        }
        map.insert(ele, ind);
    }

    None
}

pub fn check(nums: Vec<i32>) -> bool {
    //There will only be one dip in numbers
    let mut dip = 0;

    for i in 0..nums.len() {
        if nums[i] > nums[(i + 1) % nums.len()] {
            dip += 1;
        }
    }

    return if dip > 1 { false } else { true };
}

//Sort an array with 3 elements
pub fn dutch_national_flag_algorithm(arr: &mut [i32]) {
    let (mut low, mut mid, mut high) = (0, 0, arr.len() - 1);

    while mid <= high {
        match arr[mid] {
            0 => {
                arr.swap(low, mid);
                low += 1;
                mid += 1;
            }
            1 => mid += 1,
            2 => {
                arr.swap(mid, high);
                if high > 0 {
                    high -= 1;
                } else {
                    break;
                }
            }
            _ => {}
        }
    }
}

//Find the element which occurs more than n/2 times
pub fn moores_voting_algorithm(arr: &[i32]) -> Option<i32> {
    let mut leader = arr[0];
    let mut lead = 0;

    for &vote in arr {
        if lead == 0 {
            lead = 1;
            leader = vote;
        } else if vote == leader {
            lead += 1;
        } else {
            lead -= 1;
        }
    }

    if arr.iter().filter(|&&x| x == leader).count() > arr.len() / 2 {
        return Some(leader);
    }

    None
}

//Find the maximum sum possible for a subarray
pub fn kadane_algorithm(arr: &[i32]) -> i32 { //Finding the maximum subarray sum
    if arr.is_empty() {
        return 0;
    }

    let mut current = arr[0];
    let mut maximum = arr[0];

    arr.iter().skip(1).for_each(|&item| { //Skipping the first item as it is already taken
        current = std::cmp::max(item, current + item); //If the item is more than the sum, take the item
        if current > maximum {
            maximum = current;
        }
    });

    maximum
    //If we consider an empty subarray to have a sum of 0, if maximum < 0 return 0
}

pub fn find_maximum_sub_array_of_max_sum(arr: &[i32]) -> (i32, &[i32]) { //This is also Kadane's Algorithm but also finding the subarray which produces the result
    if arr.is_empty() {
        return (0, arr);
    }

    let mut current = arr[0];
    let mut maximum = arr[0];
    let mut start = 0;
    let mut end = 0;
    let mut temp_start = 0;

    for (i, &item) in arr.iter().skip(1).enumerate() {
        if item > current + item {
            current = item;
            temp_start = i + 1;
        } else {
            current += item;
        }

        if current > maximum {
            maximum = current;
            start = temp_start;
            end = i + 1;
        }
    }

    (maximum, &arr[start..=end])
}

//Find the best day to buy a stock and sell a stock
pub fn max_profit_amount(prices: &[i32]) -> i32 {
    if let Some(&first) = prices.first() {
        //fold(initial values, |(accumulator, element_of_the_array)|{function})
        let (max_profit, _) = prices.iter().skip(1).fold((0, first), |(max_profit, min_price), &price| {
            let profit = price - min_price;
            (std::cmp::max(max_profit, profit), std::cmp::min(min_price, price))
        });
        max_profit
    } else {
        0
    }
}

pub fn rearrange_array(arr: Vec<i32>) -> Vec<i32> {
    let mut ans = vec![0; arr.len()];
    let mut pos_index = 0;
    let mut neg_index = 1;

    for num in arr {
        match num.cmp(&0) {
            Ordering::Less => {
                ans[neg_index] = num;
                neg_index += 2;
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                ans[pos_index] = num;
                pos_index += 2;
            }
        }
    }

    ans
}


/*E.g. 2 1 5 4 3 0 0
To find the next permutation, we need to find the dipping point (since after 1, there is no larger permutation of 5 4 3 0 0)
Them we replace the dip with the least element on the right (this is the least element bigger than 1, 3)
Now sort the remaining elements for the least answer
*/
pub fn next_permutation(arr: &mut [i32]) {
    if arr.len() <= 1 {
        return;
    }

    let mut dip: Option<usize> = None;
    for i in (0..arr.len() - 1).rev() {
        if arr[i] < arr[i + 1] {
            dip = Some(i);
            break;
        }
    }

    match dip {
        None => {
            arr.reverse();
            return;
        }
        Some(dip) => {
            for i in (dip..arr.len()).rev() {
                if arr[i] > arr[dip] {
                    arr.swap(i, dip);
                    break;
                }
            }
            arr[dip + 1..].reverse();
        }
    }
}

pub fn longest_consecutive(nums: Vec<i32>) -> i32 {
    let set: HashSet<i32> = HashSet::from_iter(nums.iter().cloned());

    set.iter()
        .filter(|&&num| !set.get(&(num - 1)).is_some()) //If num - 1 is not in the set, num might be the start of a sequence
        .map(|&num| {
            let mut current = num;
            let mut current_longest = 1;

            while set.get(&(current + 1)).is_some() { //Now, num is always the beginning of the sequence
                current += 1;
                current_longest += 1
            }

            current_longest
        })
        .max().unwrap_or(0)
}

pub fn rotate(matrix: &mut Vec<Vec<i32>>) {
    let size = matrix.len();

    let mut swap_values = |i: usize, j: usize, ni: usize, nj: usize| {
        let temp = matrix[i][j];
        matrix[i][j] = matrix[ni][nj];
        matrix[ni][nj] = temp;
    };

    //Transposing the matrix
    for i in 0..size {
        for j in i + 1..size {
            swap_values(i, j, j, i);
        }
    }

    //Reversing the rows
    for i in matrix {
        i.reverse();
    }
}

pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
    let n = matrix.len();
    let m = matrix[0].len();

    let mut col0 = 1;
    for i in 0..n {
        for j in 0..m {
            if matrix[i][j] == 0 {
                matrix[i][0] = 0;
                if j != 0 {
                    matrix[0][j] = 0;
                } else {
                    col0 = 0;
                }
            }
        }
    }

    for i in 1..n {
        for j in 1..m {
            if matrix[i][j] != 0 {
                if matrix[0][j] == 0 || matrix[i][0] == 0 {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    if matrix[0][0] == 0 {
        for j in 0..m {
            matrix[0][j] = 0;
        }
    }
    if col0 == 0 {
        for i in 0..n {
            matrix[i][0] = 0;
        }
    }
}

pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
    let mut ans = Vec::new();
    let (rows, cols) = (matrix.len(), matrix[0].len());
    let (mut left, mut right) = (0, cols - 1);
    let (mut top, mut bottom) = (0, rows - 1);

    while top <= bottom && left <= right {
        //Left -> Right
        for i in left..=right {
            ans.push(matrix[top][i]);
        }
        top += 1;
        //Top -> Bottom
        for i in top..=bottom {
            ans.push(matrix[i][right]);
        }
        right -= 1;
        //Right -> Left
        if top <= bottom {
            for i in (left..=right).rev() {
                ans.push(matrix[bottom][i]);
            }
            bottom -= 1;
        }
        //Bottom -> Top
        if left <= top {
            for i in top..=bottom {
                ans.push(matrix[i][left]);
            }
            left += 1;
        }
    }

    ans
}

pub fn subarray_sum(arr: Vec<i32>, k: i32) -> i32 {
    let mut map = HashMap::new(); //Prefix Sum, Count
    map.insert(0, 1); //This is important for the first subarray since removing
    let mut prefix_sum = 0;
    let mut count = 0;

    for num in arr {
        prefix_sum += num;
        let extra = prefix_sum - k;

        if let Some(&val) = map.get(&extra) {
            count += val;
        }

        *map.entry(prefix_sum).or_insert(0) += 1;
    }

    count
}

pub fn pascal_triangle(rows: i32) -> Vec<Vec<i32>> {
    let mut ans = Vec::new();

    fn ncr(n: i32, r: i32) -> i32 {
        let mut ans = 1;
        for i in 0..r {
            ans *= n - i;
            ans /= i + 1;
        }
        ans
    }

    for i in 0..rows {
        let mut row = Vec::with_capacity((i + 1) as usize);
        for j in 0..=i {
            row.push(ncr(i, j));
        }
        ans.push(row);
    }

    ans
}

pub fn pascal_improved(rows: i32) -> Vec<Vec<i32>> {
    let mut triangle: Vec<Vec<i32>> = Vec::new();

    for i in 0..rows as usize {
        let mut row = vec![1; i + 1];
        if i >= 2 {
            for j in 1..i {
                row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j];
            }
        }
        triangle.push(row);
    }

    triangle
}

pub fn majority_element(arr: Vec<i32>) -> Vec<i32> {
    let (mut counter1, mut counter2) = (0, 0);
    let (mut element1, mut element2) = (i32::MIN, i32::MIN);

    for &i in &arr {
        if counter1 == 0 && i != element2 {
            counter1 = 1;
            element1 = i;
        } else if counter2 == 0 && i != element1 {
            counter2 = 1;
            element2 = i;
        } else if i == element1 {
            counter1 += 1;
        } else if i == element2 {
            counter2 += 1;
        } else {
            counter1 -= 1;
            counter2 -= 1;
        }
    }

    let mut ans = Vec::with_capacity(2);

    if arr.iter().filter(|&&x| x == element1).count() > arr.len() / 3 {
        ans.push(element1);
    }
    if arr.iter().filter(|&&x| x == element2).count() > arr.len() / 3 {
        ans.push(element2);
    }
    ans
}

pub fn three_sum(arr: Vec<i32>) -> Vec<Vec<i32>> {//Sum of three elements should be 0
    let mut arr = arr.clone();
    arr.sort();
    let mut ans = Vec::new();
    for i in 0..arr.len() {
        if i > 0 && arr[i] == arr[i - 1] {
            continue;
        }
        let (mut j, mut k) = (i + 1, arr.len() - 1);
        while j < k {
            let current_sum = arr[i] + arr[j] + arr[k];
            match current_sum.cmp(&0) {
                Ordering::Less => j += 1,
                Ordering::Equal => {
                    ans.push(vec![arr[i], arr[j], arr[k]]);
                    j += 1;
                    k -= 1;
                    while j < k && arr[j] == arr[j - 1] {
                        j += 1;
                    }
                    while j < k && arr[k] == arr[k + 1] {
                        k -= 1;
                    }
                }
                Ordering::Greater => k -= 1
            }
        }
    }
    ans
}

pub fn merge_intervals(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut intervals = intervals.clone();
    intervals.sort();
    let mut ans: Vec<Vec<i32>> = Vec::new();

    for interval in intervals {
        let a_len = ans.len(); //This is necessary since we cannot have a mutable borrow (ans.push) and an immutable borrow at the same time
        if ans.is_empty() || interval[0] > ans[a_len - 1][1] {//If the new interval is not in(side) any other interval
            ans.push(interval);
        } else {
            ans[a_len - 1][1] = std::cmp::max(ans[a_len - 1][1], interval[1]);
        }
    }

    ans
}

pub fn merge_intervals_better(intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut intervals = intervals.clone();
    intervals.sort_unstable();

    intervals.into_iter().fold(Vec::new(), |mut merged, current| {
        if let Some(last) = merged.last_mut() { //This means that the merged array is not empty
            if current[0] <= last[1] {
                last[1] = last[1].max(current[1]); //If current (interval) is inside the merged
                return merged;
            }
        }
        merged.push(current); //This is when merged is empty or current is not inside
        merged
    })
}

pub fn merge_sorted_arrays(nums1: &mut Vec<i32>, m: i32, nums2: &Vec<i32>, n: i32) {
    let mut left = m - 1;
    let mut right = 0;

    for i in 0..n {
        nums1[(m + i) as usize] = nums2[i as usize];
    }
    while left >= right {
        if left + right + 1 >= nums1.len() as i32 {
            break;
        }
        if nums1[left as usize] > nums1[(left + 1 + right) as usize] {
            nums1.swap(left as usize, (left + right + 1) as usize);
            left -= 1;
            right += 1;
        } else {
            break;
        }
    }

    nums1.sort()
}

pub fn max_product(nums: Vec<i32>) -> i32 {
    let (mut prefix, mut suffix) = (1, 1);
    let mut max = i32::MIN;

    for (&num, &rnum) in nums.iter().zip(nums.iter().rev()) {
        //If the product gets to 0, we need to reset prefix and suffix to 1 to continue
        if prefix == 0 {
            prefix = 1;
        }
        if suffix == 0 {
            suffix = 1;
        }

        prefix *= num;
        suffix *= rnum;

        max = std::cmp::max(max, std::cmp::max(prefix, suffix));
    }

    max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_largest_test() {
        assert_eq!(second_largest(&[12, 35, 1, 10, 34, 1]), Some(34));
        assert_eq!(second_largest(&[10, 5, 10]), Some(5));
        assert_eq!(second_largest(&[1, 1]), None);
    }

    #[test]
    fn two_sum_test() {
        assert_eq!(two_sum(vec![2, 7, 11, 15], 9), Some((0, 1)));
        assert_eq!(two_sum(vec![3, 2, 4], 6), Some((1, 2)));
    }

    #[test]
    fn check_test() {
        assert_eq!(check(vec![3, 4, 5, 1, 2]), true);
        assert_eq!(check(vec![2, 1, 3, 4]), false);
        assert_eq!(check(vec![1, 2, 3]), true);
    }

    #[test]
    fn dnf_test() {
        let mut arr = [2, 0, 2, 1, 1, 0];
        dutch_national_flag_algorithm(&mut arr);
        assert_eq!(arr, [0, 0, 1, 1, 2, 2]);

        let mut arr = [2, 0, 1];
        dutch_national_flag_algorithm(&mut arr);
        assert_eq!(arr, [0, 1, 2]);
    }

    #[test]
    fn moores_test() {
        assert_eq!(moores_voting_algorithm(&[3, 2, 3]), Some(3));
        assert_eq!(moores_voting_algorithm(&[2, 2, 1, 1, 1, 2, 2]), Some(2));
    }

    #[test]
    fn kadane_test() {
        assert_eq!(kadane_algorithm(&[-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6);
        assert_eq!(kadane_algorithm(&[1]), 1);
        assert_eq!(kadane_algorithm(&[5, 4, -1, 7, 8]), 23);
    }

    #[test]
    fn kadane_array() {
        let arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4];
        assert_eq!(find_maximum_sub_array_of_max_sum(&arr), (6, &arr[3..=6]));

        let arr = [1];
        assert_eq!(find_maximum_sub_array_of_max_sum(&arr), (1, &arr[0..=0]));

        let arr = [5, 4, -1, 7, 8];
        assert_eq!(find_maximum_sub_array_of_max_sum(&arr), (23, &arr[0..=4]));
    }

    #[test]
    fn stocks_test() {
        assert_eq!(max_profit_amount(&[7, 1, 5, 3, 6, 4]), 5);
        assert_eq!(max_profit_amount(&[7, 6, 4, 3, 1]), 0);
    }

    #[test]
    fn rearrange_test() {
        assert_eq!(rearrange_array(vec![3, 1, -2, -5, 2, -4]), vec![3, -2, 1, -5, 2, -4]);
        assert_eq!(rearrange_array(vec![-1, 1]), vec![1, -1]);
    }

    #[test]
    fn next_permutation_test() {
        let mut arr = [1, 2, 3];
        next_permutation(&mut arr);
        assert_eq!(arr, [1, 3, 2]);

        let mut arr = [3, 2, 1];
        next_permutation(&mut arr);
        assert_eq!(arr, [1, 2, 3]);

        let mut arr = [1, 1, 5];
        next_permutation(&mut arr);
        assert_eq!(arr, [1, 5, 1]);

        let mut arr = [2, 1, 5, 4, 3, 0, 0];
        next_permutation(&mut arr);
        assert_eq!(arr, [2, 3, 0, 0, 1, 4, 5])
    }

    #[test]
    fn longest_cons_test() {
        assert_eq!(longest_consecutive(vec![100, 4, 200, 1, 3, 2]), 4);
        assert_eq!(longest_consecutive(vec![0, 3, 7, 2, 5, 8, 4, 6, 0, 1]), 9);
    }

    #[test]
    fn rotate_matrix_test() {
        let mut matrix: Vec<Vec<i32>> = vec![
            vec![5, 1, 9, 11],
            vec![2, 4, 8, 10],
            vec![13, 3, 6, 7],
            vec![15, 14, 12, 16],
        ];
        rotate(&mut matrix);

        assert_eq!(matrix, vec![
            vec![15, 13, 2, 5],
            vec![14, 3, 4, 1],
            vec![12, 6, 8, 9],
            vec![16, 7, 10, 11],
        ]);
    }

    #[test]
    fn set_zeroes_test() {
        let mut matrix = vec![
            vec![1, 1, 1],
            vec![1, 0, 1],
            vec![1, 1, 1],
        ];
        set_zeroes(&mut matrix);
        assert_eq!(matrix, vec![
            vec![1, 0, 1],
            vec![0, 0, 0],
            vec![1, 0, 1],
        ]);

        let mut matrix = vec![
            vec![0, 1, 2, 0],
            vec![3, 4, 5, 2],
            vec![1, 3, 1, 5],
        ];
        set_zeroes(&mut matrix);
        assert_eq!(matrix, vec![
            vec![0, 0, 0, 0],
            vec![0, 4, 5, 0],
            vec![0, 3, 1, 0],
        ])
    }

    #[test]
    fn spiral_order_test() {
        assert_eq!(spiral_order(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ]), vec![1, 2, 3, 6, 9, 8, 7, 4, 5]);

        assert_eq!(spiral_order(vec![
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
            vec![9, 10, 11, 12],
        ]), vec![1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]);
    }

    #[test]
    fn subarray_sum_test() {
        assert_eq!(subarray_sum(vec![1, 1, 1], 2), 2);
        assert_eq!(subarray_sum(vec![1, 2, 3], 3), 2);
    }

    #[test]
    fn pascal_test() {
        assert_eq!(pascal_triangle(1), vec![vec![1]]);
        assert_eq!(pascal_triangle(5), vec![
            vec![1],
            vec![1, 1],
            vec![1, 2, 1],
            vec![1, 3, 3, 1],
            vec![1, 4, 6, 4, 1],
        ])
    }

    #[test]
    fn majority_element_test() {
        assert_eq!(majority_element(vec![3, 2, 3]), vec![3]);
        assert_eq!(majority_element(vec![1]), vec![1]);
        assert_eq!(majority_element(vec![4, 2, 1, 1]), vec![1]);
    }

    #[test]
    fn threesome_test() {
        assert_eq!(three_sum(vec![-1, 0, 1, 2, -1, -4]), vec![vec![-1, -1, 2], vec![-1, 0, 1]]);
        assert_eq!(three_sum(vec![0, 1, 1]), Vec::<Vec<i32>>::new());
    }

    #[test]
    fn merge_intervals_test() {
        assert_eq!(merge_intervals(vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]]), vec![vec![1, 6], vec![8, 10], vec![15, 18]]);
        assert_eq!(merge_intervals(vec![vec![1, 4], vec![4, 5]]), vec![vec![1, 5]]);
    }

    #[test]
    fn merge_intervals_better_test() {
        assert_eq!(merge_intervals_better(vec![vec![1, 3], vec![2, 6], vec![8, 10], vec![15, 18]]), vec![vec![1, 6], vec![8, 10], vec![15, 18]]);
        assert_eq!(merge_intervals_better(vec![vec![1, 4], vec![4, 5]]), vec![vec![1, 5]]);
    }

    #[test]
    fn max_product_test() {
        assert_eq!(max_product(vec![2, 3, -2, 4]), 6);
        assert_eq!(max_product(vec![-2, 0, -1]), 0);
    }
}