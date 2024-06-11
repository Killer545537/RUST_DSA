use std::collections::HashMap;

///k steps. Cards can be picked from either the end or beginning
pub fn max_score(card_points: Vec<i32>, k: usize) -> i32 {
    let n = card_points.len();
    let k = k.min(n);

    let mut lsum: i32 = card_points[..k].iter().sum();
    let mut rsum = 0;
    let mut max_sum = lsum;

    for i in (0..k).rev() {
        lsum = lsum.saturating_sub(card_points[i]);
        rsum += card_points.get(card_points.len() - k + i).unwrap_or(&0);
        max_sum = max_sum.max(lsum + rsum);
    }

    max_sum
}

pub fn length_of_longest_substring(s: String) -> usize {
    let mut char_map: HashMap<char, usize> = HashMap::new();
    let (mut l, mut r) = (0, 0);
    let mut max_len = 0;
    let s: Vec<char> = s.chars().collect();

    while r < s.len() {
        if let Some(&ind) = char_map.get(&s[r]) {
            if ind >= l {
                l = ind + 1;
            }
        }
        char_map.insert(s[r], r);
        max_len = max_len.max(r - l + 1);
        r += 1;
    }


    max_len
}

///nums contains only 0s and 1s, we can flip at most k 0s to find the most consecutive ones
pub fn longest_ones(nums: Vec<i32>, k: i32) -> usize {
    let (mut l, mut r, mut zeroes) = (0, 0, 0);
    let mut max_len = 0;

    while r < nums.len() {
        if nums[r] == 0 {
            zeroes += 1;
        }

        if zeroes > k {
            if nums[l] == 0 {
                zeroes -= 1;
            }
            l += 1;
        }

        if zeroes <= k {
            max_len = max_len.max(r - l + 1);
        }
        r += 1;
    }

    max_len
}

///We have 2 baskets and fruits[i] is a type of fruit. One basket can hold only 1 type of fruits in a sequence
pub fn total_fruits(fruits: Vec<i32>) -> usize {
    //So we need to find the max-sub-array with at most 2 different numbers
    let (mut l, mut r) = (0, 0);
    let mut max_len = 0;
    let mut freq_map = HashMap::new();

    while r < fruits.len() {
        *freq_map.entry(fruits[r]).or_insert(0) += 1;

        while freq_map.len() > 2 {
            *freq_map.get_mut(&fruits[l]).unwrap() -= 1;
            if freq_map[&fruits[l]] == 0 {
                freq_map.remove(&fruits[l]);
            }
            l += 1;
        }

        max_len = max_len.max(r - l + 1);
        r += 1;
    }

    max_len
}

///Can replace k characters to form the longest distinct sub-string
pub fn length_of_longest_substring_k_distinct(s: String, k: usize) -> usize {
    let (mut l, mut r) = (0, 0);
    let mut max_len = 0;
    let mut freq_map = HashMap::new();
    let s: Vec<char> = s.chars().collect();

    while r < s.len() {
        *freq_map.entry(s[r]).or_insert(0) += 1;

        if freq_map.len() > k {
            *freq_map.get_mut(&s[l]).unwrap() -= 1;
            if freq_map[&s[l]] == 0 {
                freq_map.remove(&s[l]);
            }
            l += 1;
        }

        if freq_map.len() <= k {
            max_len = max_len.max(r - l + 1);
        }
        r += 1;
    }

    max_len
}

///s contains only a, b and c, find the number of substrings that contain at-least one of each
pub fn number_of_substrings(s: String) -> usize {
    let mut last_seen = (None, None, None); //Index where (a,b,c) were last seen
    let mut count = 0;
    let s: Vec<char> = s.chars().collect();

    for i in 0..s.len() {
        match s[i] {
            'a' => {
                last_seen.0 = Some(i);
            }
            'b' => {
                last_seen.1 = Some(i);
            }
            'c' => {
                last_seen.2 = Some(i);
            }
            _ => {}
        }

        if last_seen.0.is_some() && last_seen.1.is_some() && last_seen.2.is_some() {
            count += (1 + std::cmp::min(std::cmp::min(last_seen.0, last_seen.1), last_seen.2).unwrap());
        }
    }

    count
}

pub fn character_replacement(s: String, k: usize) -> usize {
    let (mut l, mut r) = (0, 0);
    let s: Vec<char> = s.chars().collect();
    let mut max_len = 0;
    let mut max_frequency = 0;
    let mut freq_map = HashMap::new();

    while r < s.len() {
        *freq_map.entry(s[r]).or_insert(0) += 1;
        max_frequency = max_frequency.max(freq_map[&s[r]]);

        if (r - l + 1) - max_frequency > k { //The current sequence is not valid
            *freq_map.get_mut(&s[l]).unwrap() -= 1;
            max_frequency = freq_map.values().cloned().max().unwrap_or(0);
            l += 1;
        }

        max_len = max_len.max(r - l + 1);
        r += 1;
    }

    max_len
}

///nums is a binary array (0s and 1s)
pub fn num_subarrays_with_sum(nums: Vec<i32>, goal: i32) -> usize {
    //Number of subarrys with (sum <= goal) - (sum <= (goal-1)) = (sum == goal)
    fn helper(nums: &Vec<i32>, goal: i32) -> usize {
        if goal < 0 {
            return 0;
        }

        let (mut l, mut r) = (0, 0);
        let mut count = 0;
        let mut sum = 0;
        while r < nums.len() {
            sum += nums[r];

            while sum > goal {
                sum -= nums[l];
                l += 1;
            }

            count += r - l + 1;
            r += 1;
        }

        count
    }

    helper(&nums, goal) - helper(&nums, goal - 1)
}

///A nice sub-array is one where the number of odd integers = k
pub fn number_of_subarrays(nums: Vec<i32>, k: i32) -> i32 {
    //Convert every odd to 1 and even to 0, then find the subarray where the sum is k
    let nums: Vec<i32> = nums.iter().map(|&x| {
        if x % 2 == 0 {
            0
        } else {
            1
        }
    }).collect();

    fn helper(nums: &Vec<i32>, goal: i32) -> i32 {
        if goal < 0 {
            return 0;
        }
        let (mut l, mut r) = (0, 0);
        let mut sum = 0;
        let mut count = 0;

        while r < nums.len() {
            sum += nums[r];

            while sum > goal {
                sum -= nums[l];
                l += 1;
            }

            count += r - l + 1;
            r += 1;
        }

        count as i32
    }

    helper(&nums, k) - helper(&nums, k - 1)
}

pub fn subarrays_with_k_distinct(nums: Vec<i32>, k: usize) -> i32 {
    fn helper(nums: &Vec<i32>, k: usize) -> i32 {
        let (mut l, mut r) = (0, 0);
        let mut count = 0;
        let mut freq_map = HashMap::new();

        while r < nums.len() {
            *freq_map.entry(nums[r]).or_insert(0) += 1;

            while freq_map.len() > k {
                *freq_map.get_mut(&nums[l]).unwrap() -= 1;

                if freq_map[&nums[l]] == 0 {
                    freq_map.remove(&nums[l]);
                }

                l += 1;
            }

            count += r - l + 1; //If an array is possible, then so are it sub-arrays
            r += 1;
        }

        count as i32
    }

    helper(&nums, k) - helper(&nums, k - 1)
}

///Return the min sub-string which contains every character in t
pub fn min_window(s: String, t: String) -> String {
    let s: Vec<char> = s.chars().collect();
    let t: Vec<char> = t.chars().collect();
    let mut min_len = usize::MAX;
    let mut start_index = None;
    let mut count = 0;
    let mut freq_map = HashMap::new();

    for &c in t.iter() {
        *freq_map.entry(c).or_insert(0) += 1;
    }

    let (mut l, mut r) = (0, 0);

    for (i, &c) in s.iter().enumerate() {
        match freq_map.get_mut(&c) {
            Some(val) => {
                if *val > 0 {
                    count += 1;
                }
                *val -= 1;
            }
            None => {}
        }

        while count == t.len() {
            if i - l + 1 < min_len {
                min_len = i - l + 1;
                start_index = Some(l);
            }

            match freq_map.get_mut(&s[l]) {
                Some(val) => {
                    if *val == 0 {
                        count -= 1;
                    }
                    *val += 1;
                }
                None => {}
            }
            l += 1;
        }

        r = i;
    }

    start_index.map_or(String::new(), |start| s[start..start + min_len].iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_score_test() {
        assert_eq!(max_score(vec![1, 2, 3, 4, 5, 6, 1], 3), 12);
        assert_eq!(max_score(vec![2, 2, 2], 2), 4);
        assert_eq!(max_score(vec![9, 7, 7, 9, 7, 7, 9], 7), 55);
    }

    #[test]
    fn length_of_longest_substring_test() {
        assert_eq!(length_of_longest_substring("abcabcbb".to_string()), 3);
        assert_eq!(length_of_longest_substring("bbbbb".to_string()), 1);
    }

    #[test]
    fn longest_ones_test() {
        assert_eq!(longest_ones(vec![1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2), 6);
        assert_eq!(longest_ones(vec![0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], 3), 10);
    }

    #[test]
    fn fruit_test() {
        assert_eq!(total_fruits(vec![2, 1, 2]), 3);
        assert_eq!(total_fruits(vec![0, 1, 2, 2, 2, 2]), 5);
    }

    #[test]
    fn length_of_longest_substring_k_test() {
        assert_eq!(length_of_longest_substring_k_distinct("eceba".to_string(), 2), 3);
        assert_eq!(length_of_longest_substring_k_distinct("aa".to_string(), 1), 2);
    }

    #[test]
    fn number_of_substrings_test() {
        assert_eq!(number_of_substrings("abcabc".to_string()), 10);
        assert_eq!(number_of_substrings("aaacb".to_string()), 3);
        assert_eq!(number_of_substrings("abc".to_string()), 1);
    }

    #[test]
    fn character_replacement_test() {
        assert_eq!(character_replacement("ABAB".to_string(), 2), 4);
        assert_eq!(character_replacement("AABABBA".to_string(), 1), 4);
    }

    #[test]
    fn num_subarrays_with_sum_test() {
        assert_eq!(num_subarrays_with_sum(vec![1, 0, 1, 0, 1], 2), 4);
        assert_eq!(num_subarrays_with_sum(vec![0, 0, 0, 0, 0], 0), 15);
    }

    #[test]
    fn nice_test() {
        assert_eq!(number_of_subarrays(vec![1, 1, 2, 1, 1], 3), 2);
        assert_eq!(number_of_subarrays(vec![2, 4, 6], 1), 0);
    }

    #[test]
    fn subarrays_with_k_distinct_test() {
        assert_eq!(subarrays_with_k_distinct(vec![2, 1, 1, 1, 3, 4, 3, 2], 3), 9);
        assert_eq!(subarrays_with_k_distinct(vec![1, 2, 1, 2, 3], 2), 7);
        assert_eq!(subarrays_with_k_distinct(vec![1, 2, 1, 3, 4], 3), 3);
    }

    #[test]
    fn min_window_test() {
        assert_eq!(min_window("ADOBECODEBANC".to_string(), "ABC".to_string()), "BANC".to_string());
    }
}