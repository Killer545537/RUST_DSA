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

pub fn find_min(arr: &[i32]) -> i32 {
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
}