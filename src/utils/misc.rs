pub fn elementwise_addition<T>(x: &Vec<T>, y: &Vec<T>) -> Vec<T>
where
    T: std::ops::Add + Copy,
    Vec<T>: FromIterator<<T as std::ops::Add>::Output>,
{
    x.iter().zip(y.iter()).map(|(x, y)| *x + *y).collect()
}
