use std::cmp::{Ordering};

//Rust has a built-in function arr.binary_search() -> Result<Index if found, Index it should be inserted at if not>
pub fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let (mut low, mut high) = (0, arr.len());

    while low < high {
        let mid = (low + high) / 2;

        match target.cmp(&arr[mid]) {
            Ordering::Less => {//If the target is less than mid, then it must be in the first half
                high = mid;
            }
            Ordering::Equal => {
                return Some(mid);
            }
            Ordering::Greater => {//If the target is more than mid, then it must be in the second half
                low = mid + 1;
            }
        }
    }

    None
}

//It is the smallest index such that arr[i] >= target, i.e. it is the index of the smallest number greater than or equal to the target
//This is also called ceil, but upper_bound IS NOT floor
pub fn lower_bound(arr: &[i32], target: i32) -> Option<usize> { //This is also the find search_index problem
    let (mut low, mut high) = (0, arr.len());

    while low < high {
        let mid = (low + high) / 2;

        match arr[mid].cmp(&target) {
            Ordering::Greater | Ordering::Equal => high = mid,
            Ordering::Less => low = mid + 1,
        }
    }

    if high == arr.len() {
        None
    } else {
        Some(high)
    }
}

//It is the smallest index 'i' such that arr[i] > target, i.e. it is the index of the smallest number greater than the target
pub fn upper_bound(arr: &[i32], target: i32) -> Option<usize> {
    let (mut low, mut high) = (0, arr.len());

    while low < high {
        let mid = (low + high) / 2;
        match arr[mid].cmp(&target) {
            Ordering::Greater => high = mid,
            _ => low = mid + 1
        }
    }

    if high == arr.len() {
        None
    } else {
        Some(high)
    }
}

pub fn floor(arr: &[i32], target: i32) -> Option<usize> {
    let (mut low, mut high) = (0, arr.len());

    while low < high {
        let mid = (low + high) / 2;

        match arr[mid].cmp(&target) {
            Ordering::Less | Ordering::Equal => low = mid + 1,
            Ordering::Greater => high = mid
        }
    }

    if high == 0 {
        None
    } else {
        Some(high - 1)
    }
}


pub fn search_range(arr: &[i32], target: i32) -> Option<(usize, usize)> {//This is equivalent to finding the lower bound (1st occurrence) and the upper bound - 1 (last occurrence)
    //This is also the number of occurrences of target in the array
    let first_occurrence = lower_bound(arr, target);
    let last_occurrence = upper_bound(arr, target);

    //If the lower bound is not equal to the target, then it is not in the array
    return if let Some(first) = first_occurrence {
        if arr[first] != target {
            None
        } else {
            Some((first, last_occurrence.unwrap_or(arr.len()) - 1)) //If last_occurrence is None it means that target is the last element
        }
    } else {
        None
    };
}

/*
pub fn search_range(arr: &[i32], target: i32) -> Option<(usize, usize)> {
    if let Ok(index) = arr.binary_search(&target) {
        Some((arr.partition_point(|&x| x < target), arr.partition_point(|&x| x <= target) - 1))
    } else {
        None
    }
}
 */

pub fn search_sorted_unique(arr: &[i32], target: i32) -> Option<usize> {
    let (mut low, mut high) = (0, arr.len() - 1);

    while low <= high {
        let mid = (low + high) / 2;

        if arr[mid] == target {
            return Some(mid);
        }
        //If the left half is sorted
        if arr[low] <= arr[mid] {
            if arr[low] <= target && target <= arr[mid] {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        } else { //If the right half is sorted
            if arr[mid] <= target && target <= arr[high] {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }

    None
}

pub fn search_sorted(arr: &[i32], target: i32) -> bool {
    //This is essentially the same problem except the case when arr[low] = arr[mid] = arr[high]
    // In that case we cannot find the sorted portion
    let (mut low, mut high) = (0, arr.len() - 1);

    while low <= high {
        let mid = (low + high) / 2;

        if arr[mid] == target {
            return true;
        }
        if arr[low] == arr[mid] && arr[mid] == arr[high] {
            low += 1;
            high = high.checked_sub(1).unwrap_or(0);
            continue;
        }

        if arr[low] <= arr[mid] {
            if arr[low] <= target && target <= arr[mid] {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        } else {
            if arr[mid] <= target && target <= arr[high] {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }

    false
}

pub fn find_min(arr: &[i32]) -> i32 { //This is also the number of times the array has been rotated to the right
    //The unsorted portion of the array will always have the min, except in the case when mid is the minimum
    let (mut low, mut high) = (0, arr.len() - 1);
    let mut min = i32::MAX;

    while low <= high {
        let mid = (low + high) / 2;

        if arr[low] <= arr[mid] {//Left portion of the array is sorted
            min = min.min(arr[low]);
            low = mid + 1;
        } else {
            min = min.min(arr[mid]); //EDGE CASE handling
            high = mid - 1;
        }
    }

    min
}

pub fn single_non_duplicate(arr: &[i32]) -> Option<i32> { //Every element in the array appears twice except one
    if arr.len() == 1 {
        return Some(arr[0]);
    }
    //If the first element is the single element
    if arr[0] != arr[1] {
        return Some(arr[0]);
    }
    //If the last element is the single element
    if arr[arr.len() - 1] != arr[arr.len() - 2] {
        return Some(arr[arr.len() - 1]);
    }
    //The first and last are handled separately since they are edge cases in the binary search (if mid == first/last out of index issues)
    let (mut low, mut high) = (1, arr.len() - 2);

    while low <= high {
        let mid = (low + high) / 2;

        if arr[mid - 1] != arr[mid] && arr[mid] != arr[mid + 1] {
            return Some(arr[mid]);
        }
        //On the left half pairs are in (odd, even) indices, so the ans is on the left half
        if (mid % 2 == 0 && arr[mid] == arr[mid + 1]) || (mid % 2 == 1 && arr[mid - 1] == arr[mid]) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    None
}

//The peak element is such that arr[i-1] < arr[i] > arr[i+1]
pub fn find_peak_element(arr: &[i32]) -> Option<usize> {
    //Here, we can assume that arr[-1] = arr[n] = -INFINITY
    if arr.len() == 1 {
        return Some(0);
    }
    //If the first element is the peak
    if arr[0] > arr[1] {
        return Some(0);
    }
    //If the last element is the peak
    if arr[arr.len() - 1] > arr[arr.len() - 2] {
        return Some(arr.len() - 1);
    }

    let (mut low, mut high) = (1, arr.len() - 2);

    while low <= high {
        let mid = (low + high) / 2;

        if arr[mid - 1] < arr[mid] && arr[mid] > arr[mid + 1] {
            return Some(mid);
        } else if arr[mid] > arr[mid - 1] {//Mid is on the increasing curve
            low = mid + 1;
        } else if arr[mid] > arr[mid + 1] {
            high = mid - 1;
        } else {//This is case where mid is at the trough (opposite of peak)
            low = mid + 1; //Or high = mid - 1, since both the halves will have a peak
        }
    }

    None
}

pub fn find_sqrt(n: i32) -> i32 {
    let (mut low, mut high) = (0, n);

    while low <= high {
        let mid = (low + high) / 2;
        if mid * mid <= n {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    high
}

pub fn find_nth_root(n: i32, k: i32) -> i32 { //n√k
    let (mut low, mut high) = (1, k);

    fn power(mut x: i32, mut n: i32) -> i32 { //x^n
        let mut ans = 1;

        while n > 0 {
            if n % 2 == 1 {
                ans *= x;
                n -= 1;
            } else {
                n /= 2;
                x *= x;
            }
        }
        ans
    }

    while low <= high {
        let mid = (low + high) / 2;
        let mid_pow = power(mid, n);

        if mid_pow == k {
            return mid;
        } else if mid_pow < k {
            low = mid + 1;
        } else {
            high = high - 1;
        }
    }

    -1
}

//We need to find the lowest possible bananas/hr so that all bananas are eaten
// However, if a pile is finished in less than 1 complete hour, the rest of the time will be wasted (if 3 bananas and 2 banana/hr, then 2hr taken
pub fn min_eating_speed(piles: &[i64], hours: i64) -> i64 { //Using i64 just to avoid overflow conditions
    //The maximum time taken will be when 1 banana/hr and min when max(piles)/hr
    let (mut low, mut high) = (1, *piles.iter().max().unwrap_or(&1));

    let hours_taken = |speed: i64| -> i64 {
        piles.iter().map(|&pile| (pile + speed - 1) / speed).sum()
    };

    while low <= high {
        let mid = (low + high) / 2;

        if hours_taken(mid) <= hours {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    low
}

pub fn min_days_bouquets(bloom_days: &[i32], bouquets: i32, adjacent: i32) -> Option<i32> {
    if (bloom_days.len() as i32) < (bouquets * adjacent) {
        return None;
    }

    let number_of_bouquets = |day: i32| -> i32{
        let mut streak = 0;
        let mut ans = 0; //This is the number of bouquets that can be formed

        for &bloom in bloom_days {
            if bloom <= day {
                streak += 1;
            } else {
                ans += streak / adjacent;
                streak = 0;
            }
        }
        ans += streak / adjacent;
        ans
    };

    let (mut low, mut high) = (*bloom_days.iter().min().unwrap_or(&1), *bloom_days.iter().max().unwrap_or(&1));

    while low <= high {
        let mid = (low + high) / 2;

        if number_of_bouquets(mid) >= bouquets {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    Some(low)
}

pub fn smallest_divisor(arr: &[i32], threshold: i32) -> i32 {
    let (mut low, mut high) = (1, *arr.iter().max().unwrap());

    let division_sum = |divisor: i32| {
        let mut sum: i32 = 0;

        for &i in arr {
            match sum.checked_add((i + divisor - 1) / divisor) {
                Some(new_sum) => sum = new_sum,
                None => return i32::MAX,
            }
        }

        sum
    };

    while low <= high {
        let mid = (low + high) / 2;

        if division_sum(mid) > threshold {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    low
}

pub fn ship_within_days(weights: &[i32], days: i32) -> i32 {
    //The days taken != ceil(total/capacity) since the weights are not equally distributed and cannot be rearranged
    let days_taken = |capacity: i32| -> i32 {
        weights.iter().fold((0, 1), |(total, days), &weight| {
            if total + weight > capacity {
                (weight, days + 1)
            } else {
                (total + weight, days)
            }
        }).1
    };

    let (mut low, mut high) = (*weights.iter().max().unwrap(), weights.iter().sum());

    while low <= high {
        let mid = (low + high) / 2;
        match days.cmp(&days_taken(mid)) {
            Ordering::Greater | Ordering::Equal => high = mid - 1, //If fewer days are being taken, the answer is on the left
            Ordering::Less => low = mid + 1,
        }
    }

    low
}

pub fn find_kth_positive(arr: &[i32], k: i32) -> i32 {
    let (mut low, mut high) = (0, arr.len());

    while low < high {
        let mid = (low + high) / 2;
        /*  If all the numbers were present the array would look like ->
        Index -> 0 1 2 3 4 5 6
        Ele   -> 1 2 3 4 5 6 7 (i + 1)
        So, the number of missing elements till an index 'i' is arr[i] - (i + 1)  */
        let missing = arr[mid] - (mid as i32 + 1);

        if missing < k {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    /*  Now, after finding the high, the kth missing number will be,
    arr[high] + more (more is the number of missing left after high)
    more = k - missing
    arr[high] + more = arr[high] + k - missing = high + 1 + k  = low + k */
    low as i32 + k
}

pub fn find_kth_positive_simpler(arr: &[i32], k: i32) -> i32 {
    (1..)
        .filter(|x| arr.binary_search(x).is_err())
        .nth(k as usize - 1)
        .unwrap()
}

//In this problem we need to place the cows in the stalls
//In any given configuration we can find the distance between any two cows as stalls[i] - stalls[j]
//We need to find the max(minimum distance) in a given configuration
//Good way to remember how to return low/high, if low is possible at the end it will switch and point to an impossible case and similarly for high
pub fn aggressive_cows(stalls: &[i32], cows: i32) -> i32 {
    let mut stalls = stalls.to_vec();
    stalls.sort();
    let (mut low, mut high) = (1, *stalls.last().unwrap() - *stalls.first().unwrap());
    let possible_distance = |distance: i32| -> bool {
        let mut number_cows = 1;
        let mut last_stall = stalls[0];

        for &stall in stalls.iter() {
            if stall - last_stall >= distance {
                number_cows += 1;
                last_stall = stall;
            }
        }

        return if number_cows >= cows {
            true
        } else {
            false
        };
    };

    while low <= high {
        let mid = (low + high) / 2;

        if possible_distance(mid) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    high
}

//Here, each student must be given at least one book (if more, then in a contiguous manner)
//Allocate books in such a way that the maximum number of pages given to a student is minimum, min(max(pages to a student))
pub fn allocate_books(books: &[i32], students: i32) -> Option<i32> {
    //The only case when it is not possible to allocate books,
    if students > books.len() as i32 {
        return None;
    }

    let (mut low, mut high) = (*books.iter().max().unwrap(), books.iter().sum());
    let students_allocated = |pages: i32| -> i32 {//pages -> maximum pages a student can read
        books.iter().fold((1, 0), |(students, allocated), &book| {
            if allocated + book <= pages {
                (students, allocated + book)
            } else {
                (students + 1, book)
            }
        }).0
    };

    while low <= high {
        let mid = (low + high) / 2;

        if students_allocated(mid) > students {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    Some(low)
}

//Here, we have two painters which take arr[i] time to do a task, we need to find the min of the maximum time taken (take only contiguous sub-arrays as tasks)
pub fn painter_partition(arr: &[i32]) -> Option<i32> {
    arr.iter().enumerate()
        .map(|(i, _)| std::cmp::max(arr[..=i].iter().sum(), arr[i + 1..].iter().sum()))
        .min()
}

//Here, we need to split the array into k parts that the maximum sub-array is minimum (empty sub-arrays are not allowed)
//Exactly the same problem as allocate_books
pub fn split_array(arr: &[i32], k: i32) -> i32 {
    let partitions = |possible_sum: i32| -> i32 {
        arr.iter().fold((1, 0), |(parts, sum), &current| {
            if sum + current <= possible_sum {
                (parts, sum + current)
            } else {
                (parts + 1, current)
            }
        }).1
    };

    let (mut low, mut high) = (*arr.iter().max().unwrap(), arr.iter().sum());

    while low <= high {
        let mid = (low + high) / 2;

        if partitions(mid) > k {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    low
}

pub fn minimize_max_distance(coordinates: &[f32], new_stations: i32) { //The new gas stations can be placed between two coordinates such that it is fractional
    //Clearly, it is always disadvantageous to add new coordinates to the left or right
    //If we need to place one station, we should place it in the largest gap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_search_test() {
        let arr = [1, 2, 3, 4];
        assert_eq!(binary_search(&arr, &1), Some(0));
        assert_eq!(binary_search(&arr, &10), None);
    }

    #[test]
    fn lower_bound_test() {
        assert_eq!(lower_bound(&[1, 3, 4, 4, 10], 4), Some(2));
        assert_eq!(lower_bound(&[1, 2, 8, 10, 11, 12, 19], 5), Some(2));
        assert_eq!(lower_bound(&[1, 3, 8, 9], 10), None);
        assert_eq!(lower_bound(&[4, 5, 6, 7], 2), Some(0));
    }

    #[test]
    fn upper_bound_test() {
        assert_eq!(upper_bound(&[5, 6, 8, 8], 7), Some(2))
    }

    #[test]
    fn floor_test() {
        assert_eq!(floor(&[10, 20, 30, 40], 25), Some(1));
        assert_eq!(floor(&[10, 20, 30, 40], 100), Some(3));
        assert_eq!(floor(&[3, 4, 7, 8, 8, 10], 6), Some(1));
        assert_eq!(floor(&[10, 20, 30, 40], 5), None);
    }

    #[test]
    fn search_range_test() {
        assert_eq!(search_range(&[5, 7, 7, 8, 8, 10], 8), Some((3, 4)));
        assert_eq!(search_range(&[5, 7, 7, 8, 8, 10], 6), None);
        assert_eq!(search_range(&[], 0), None);
        assert_eq!(search_range(&[1], 1), Some((0, 0)));
    }

    #[test]
    fn search_sorted_unique_test() {
        assert_eq!(search_sorted_unique(&[4, 5, 6, 7, 0, 1, 2], 0), Some(4));
        assert_eq!(search_sorted_unique(&[4, 5, 6, 7, 0, 1, 2], 3), None);
        assert_eq!(search_sorted_unique(&[1], 0), None);
        assert_eq!(search_sorted_unique(&[3, 1], 2), None);
    }

    #[test]
    fn search_sorted_test() {
        assert_eq!(search_sorted(&[2, 5, 6, 0, 0, 1, 2], 3), false);
    }

    #[test]
    fn single_non_duplicate_test() {
        assert_eq!(single_non_duplicate(&[1, 1, 2, 3, 3, 4, 4, 8, 8]), Some(2));
        assert_eq!(single_non_duplicate(&[3, 3, 7, 7, 10, 11, 11]), Some(10));
    }

    #[test]
    fn find_peak_test() {
        assert_eq!(find_peak_element(&[1, 2, 3, 1]), Some(2));
        assert_eq!(find_peak_element(&[1, 2, 1, 3, 5, 6, 4]), Some(5));
        assert_eq!(find_peak_element(&[1, 2, 1, 2, 1]), Some(3));
    }

    #[test]
    fn sqrt_test() {
        assert_eq!(find_sqrt(0), 0);
        assert_eq!(find_sqrt(1), 1);
        assert_eq!(find_sqrt(16), 4);
        assert_eq!(find_sqrt(21), 4);
    }

    #[test]
    fn eating_test() {
        assert_eq!(min_eating_speed(&[3, 6, 7, 11], 8), 4);
        assert_eq!(min_eating_speed(&[30, 11, 23, 4, 20], 5), 30);
        assert_eq!(min_eating_speed(&[30, 11, 23, 4, 20], 6), 23);
        assert_eq!(min_eating_speed(&[805306368, 805306368, 805306368], 1000000000), 3);
    }

    #[test]
    fn bouquets_test() {
        assert_eq!(min_days_bouquets(&[1, 10, 3, 10, 2], 3, 1), Some(3));
        assert_eq!(min_days_bouquets(&[1, 10, 3, 10, 2], 3, 2), None);
    }

    #[test]
    fn smallest_divisor_test() {
        assert_eq!(smallest_divisor(&[1, 2, 5, 9], 6), 5);
        assert_eq!(smallest_divisor(&[44, 22, 33, 11, 1], 5), 44);
        assert_eq!(smallest_divisor(&[21212, 10101, 12121], 1000000), 1);
    }

    #[test]
    fn shipper_test() {
        assert_eq!(ship_within_days(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5), 15);
        assert_eq!(ship_within_days(&[3, 2, 2, 4, 1, 4], 3), 6);
        assert_eq!(ship_within_days(&[1, 2, 3, 1, 1], 4), 3);
    }

    #[test]
    fn missing_test() {
        assert_eq!(find_kth_positive(&[2], 1), 1);
        assert_eq!(find_kth_positive(&[2, 3, 4, 7, 11], 5), 9);
        assert_eq!(find_kth_positive(&[1, 2, 3, 4], 2), 6);
    }

    #[test]
    fn missing_simpler_test() {
        assert_eq!(find_kth_positive_simpler(&[2], 1), 1);
    }

    #[test]
    fn aggressive_test() {
        assert_eq!(aggressive_cows(&[1, 2, 3, 4, 7], 3), 3);
        assert_eq!(aggressive_cows(&[5, 4, 3, 2, 1, 1000000000], 2), 999999999);
    }

    #[test]
    fn allocation_test() {
        assert_eq!(allocate_books(&[12, 34, 67, 90], 2), Some(113));
    }
}