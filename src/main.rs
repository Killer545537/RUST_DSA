#![allow(dead_code)]

mod arrays;

fn main() {
    let mut vec = vec![vec![1,2,3], vec![4,5,6]];

    for i in vec.iter() {
        println!("{i:?}");
    }

    println!("{vec:?}")
}