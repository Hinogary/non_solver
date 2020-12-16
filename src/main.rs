use derive_more::Display as macro_Display;
use itertools::Itertools;
use rayon::prelude::*;
use std::fs;
use std::str::FromStr;
use std::time::{Duration, Instant};
use structopt::StructOpt;
use std::fs::File;
use std::io::LineWriter;
use std::io::prelude::*;

#[derive(Debug)]
enum SquareMatrix {
    SparseCSR(Vec<f64>, Vec<usize>, Vec<usize>),
    Full(Vec<f64>, usize),
}

impl SquareMatrix {
    fn multiply_vec_into(&self, vec: &Vector, target: &mut Vector) {
        assert!(
            target.size() == vec.size() && target.size() == self.size(),
            format!("{} {} {}", self.size(), vec.size(), target.size())
        );
        match self {
            SquareMatrix::SparseCSR(values, indcol, indrow) => indrow
                .iter()
                .zip(indrow.iter().skip(1)) // two adjanced elements are returned
                .zip(target.0.iter_mut())
                .for_each(|((&f, &t), tar)| {
                    *tar = values[f..t]
                        .iter()
                        .zip(indcol[f..t].iter().map(|&v_i| vec.0[v_i]))
                        .map(|(m, v)| m * v)
                        .sum();
                }),
            SquareMatrix::Full(values, n) => {
                target
                    .0
                    .iter_mut()
                    .zip(values.chunks(*n))
                    .for_each(|(tar, chunk)| {
                        *tar = chunk.iter().zip(vec.0.iter()).map(|(&m, &v)| m * v).sum()
                    })
            }
        }
    }
    fn size(&self) -> usize {
        match self {
            SquareMatrix::Full(_, n) => *n,
            SquareMatrix::SparseCSR(_, _, indrow) => indrow.len() - 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Vector(Vec<f64>);



impl Vector {
    fn dot(&self) -> f64 {
        self.0.iter().map(|x| x * x).sum()
    }
    fn size(&self) -> usize {
        self.0.len()
    }
    fn zeros(n: usize) -> Self {
        Vector(vec![0.0; n])
    }
    fn minus(self) -> Self {
        Vector(self.0.into_iter().map(|x| -x).collect())
    }
    fn dot_with(&self, other: &Self) -> f64 {
        self.0.iter().zip(other.0.iter()).map(|(x, y)| x * y).sum()
    }
    fn add_coeff_vector(&mut self, coeff: f64, other: &Vector) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(x, y)| *x += coeff * y)
    }
    fn sub_coeff_vector(&mut self, coeff: f64, other: &Vector) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(x, y)| *x -= coeff * y)
    }
}

impl std::ops::AddAssign<&Vector> for Vector {
    fn add_assign(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(lhs, rhs)| *lhs += rhs)
    }
}

impl std::ops::SubAssign<&Vector> for Vector {
    fn sub_assign(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(lhs, rhs)| *lhs -= rhs)
    }
}

impl std::ops::MulAssign<f64> for Vector {
    fn mul_assign(&mut self, coeff: f64) {
        self.0.iter_mut().for_each(|x| *x *= coeff);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_dense_mat_mul() {
        let mat = super::SquareMatrix::Full(vec![2.0, 4.0, 8.0, 16.0], 2);
        let vec = super::Vector(vec![1.0, 2.0]);
        let mut dest = super::Vector::zeros(2);
        mat.multiply_vec_into(&vec, &mut dest);
        assert!(dest == super::Vector(vec![10., 40.]));
    }

    #[test]
    fn test_sparse_mat_mul() {
        let mat = super::SquareMatrix::SparseCSR(
            vec![2.0, 4.0, 8.0, 16.0],
            vec![0, 1, 0, 1],
            vec![0, 2, 4],
        );
        let vec = super::Vector(vec![1.0, 2.0]);
        let mut dest = super::Vector::zeros(2);
        mat.multiply_vec_into(&vec, &mut dest);
        assert!(dest == super::Vector(vec![10., 40.]));
    }

    #[test]
    fn test_sub_assign() {
        let vec1 = super::Vector(vec![1., 2.]);
        let mut vec2 = super::Vector(vec![2., -4.]);
        vec2 -= &vec1;
        assert!(vec2 == super::Vector(vec![1., -6.]));
    }

    #[test]
    fn test_add_assign() {
        let vec1 = super::Vector(vec![1., 2.]);
        let mut vec2 = super::Vector(vec![2., 4.]);
        vec2 += &vec1;
        assert!(vec2 == super::Vector(vec![3., 6.]));
    }
}

fn main() -> Result<(), DisplayError> {
    let opts = Opts::from_args();
    let mut x = Vector::zeros(opts.matrix.0.size());
    let matrix = &opts.matrix.0;
    let rhs = &opts.rhs.0;
    let epsilon = opts.epsilon;
    match opts.method.as_str() {
        "combined_gradient" => combined_gradient_method(matrix, &mut x, rhs, epsilon),
        "gradient" => gradient_method(matrix, &mut x, rhs, epsilon),
        _ => {
            return Err(DisplayError(format!(
                "Method must be in {{gradient, combined_gradient}}, got: {}",
                opts.method
            )))
        }
    };
    //println!("{:?}", x);
    let mut file = File::create("matrix.txt").unwrap();
    let mut file = LineWriter::new(file);
    for val in x.0 {
        file.write_all(format!("{}\n", val).as_bytes());
    }

    Ok(())
}

fn gradient_method(A: &SquareMatrix, x: &mut Vector, b: &Vector, epsilon: f64) -> f64 {
    let start = Instant::now();
    let n = b.size();
    let mut residium = Vector::zeros(n);
    let mut Aresidium = Vector::zeros(n);
    A.multiply_vec_into(x, &mut residium);
    residium -= b;
    residium = residium.minus();
    let epsilon_squared = epsilon * epsilon;
    let mut residium_dot = residium.dot();
    let eps = (1..)
        .map(|i| {
            let start = Instant::now();

            A.multiply_vec_into(&residium, &mut Aresidium);
            let alpha = residium_dot / residium.dot_with(&Aresidium);
            x.add_coeff_vector(alpha, &residium);
            residium.sub_coeff_vector(alpha, &Aresidium);
            residium_dot = residium.dot();

            let end = Instant::now();
            if true {
                println!("{}: {}, {:?}", i, residium_dot.sqrt(), end - start)
            }
            residium_dot
        })
        .find_map(|err_squared| {
            if err_squared < epsilon_squared {
                Some(err_squared.sqrt())
            } else {
                None
            }
        })
        .unwrap();
    let end = Instant::now();
    println!("Duration {:?}", end - start);
    eps
}

fn combined_gradient_method(A: &SquareMatrix, x: &mut Vector, b: &Vector, epsilon: f64) -> f64 {
    let start = Instant::now();
    let n = b.size();
    let mut residium = Vector::zeros(n);
    let mut Asomevec = Vector::zeros(n);
    A.multiply_vec_into(x, &mut residium);
    residium -= b;
    residium = residium.minus();
    let mut somevec = residium.clone();
    let epsilon_squared = epsilon * epsilon;
    let mut residium_dot = residium.dot();
    let eps = (0..)
        .map(|i| {
            let start = Instant::now();

            A.multiply_vec_into(&somevec, &mut Asomevec);
            let alpha = residium_dot / somevec.dot_with(&Asomevec);
            x.add_coeff_vector(alpha, &somevec);
            residium.sub_coeff_vector(alpha, &Asomevec);
            let new_residium_dot = residium.dot();
            let beta = new_residium_dot / residium_dot;
            residium_dot = new_residium_dot;
            somevec *= beta;
            somevec += &residium;

            let end = Instant::now();
            if true {
                println!("{}: {}, {:?}", i, residium_dot.sqrt(), end - start)
            }
            new_residium_dot
        })
        .find_map(|err_squared| {
            if err_squared <= epsilon_squared {
                Some(err_squared.sqrt())
            } else {
                None
            }
        })
        .unwrap();
    let end = Instant::now();
    println!("Duration {:?}", end - start);
    eps
}

#[derive(macro_Display, Debug)]
#[display(fmt="{}", self.0)]
pub struct DisplayError(String);

#[derive(Debug)]
struct MatrixFromFile(SquareMatrix);

impl FromStr for MatrixFromFile {
    type Err = DisplayError;
    fn from_str(file_name: &str) -> Result<Self, Self::Err> {
        let file_content = fs::read_to_string(file_name).map_err(|e| {
            DisplayError(format!(
                "Could not load matrix file: {}, because: {}",
                file_name, e
            ))
        })?;
        let mut lines = file_content.lines();
        let first_line = lines.next().unwrap();
        let first_line = first_line
            .split(' ')
            .filter(|x| !x.is_empty())
            .map(|x| {
                x.parse().map_err(|e| {
                    DisplayError(format!(
                        "Failed to parse number: {} in file: {} because: {}",
                        x, file_name, e
                    ))
                })
            })
            .collect::<Result<Vec<usize>, _>>()?;
        match &first_line[..] {
            &[n, numbers/*there is how many numbers are in file ... ignore it*/] => {
                let mut triples = lines
                    .flat_map(|line| line.split(' ')
                    .filter(|x| !x.is_empty() && *x != "\n"))
                    .tuples()
                    .map(|(i, j, value)|
                        Ok((
                            i.parse().map_err(|e|DisplayError(format!("Failed to parse number: {}, because: {}", i, e)))?,
                            j.parse().map_err(|e|DisplayError(format!("Failed to parse number: {}, because: {}", j, e)))?,
                            value.parse().map_err(|e|DisplayError(format!("Failed to parse number: {}, because: {}", value, e)))?
                        ))
                    )
                    .collect::<Result<Vec<(usize, usize, f64)>,_>>()?;
                if triples.len() != numbers{
                    return Err(DisplayError(format!("Wrong number of floats in matrix file: {}", file_name)))
                }
                triples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let mut values = Vec::with_capacity(numbers);
                let mut indcols = Vec::with_capacity(numbers);
                let mut indrow = Vec::with_capacity(n+1);
                indrow.push(0);
                let mut last_i = 0;
                for (index, (i, j, value)) in triples.into_iter().enumerate(){
                    values.push(value);
                    indcols.push(j);
                    while last_i < i {
                        indrow.push(index);
                        last_i += 1;
                    }
                }
                indrow.push(numbers);
                Ok(MatrixFromFile(SquareMatrix::SparseCSR(values, indcols, indrow)))
            }
            &[n] => {
                let values = lines.flat_map(|line| line.split(' ').filter(|x| !x.is_empty() && *x != "\n")).map(|x| x.parse().map_err(|e|DisplayError(format!(
                    "Failed to parse number: {} in file: {} because: {}",
                    x, file_name, e
                )))).collect::<Result<Vec<f64>,_>>()?;
                if values.len() != n * n {
                    return Err(DisplayError(format!("There is incorrect count of numbers in file. Expected: {} Got: {}", n*n, values.len())));
                }
                Ok(MatrixFromFile(SquareMatrix::Full(values, n)))
            }
            _ => Err(DisplayError(format!("Bad matrix format in file: {}", file_name)))
        }
    }
}

#[derive(Debug)]
struct VectorFromFile(Vector);

impl FromStr for VectorFromFile {
    type Err = DisplayError;
    fn from_str(file_name: &str) -> Result<Self, Self::Err> {
        Ok(VectorFromFile(Vector(
            fs::read_to_string(file_name)
                .map_err(|e| {
                    DisplayError(format!(
                        "Could not load vector file: {}, because: {}",
                        file_name, e
                    ))
                })?
                .lines()
                .filter(|&x| x.find('.').is_some()) // this works on all lines, first would be enough ... whatever
                .map(|x| {
                    x.trim().parse().map_err(|e| {
                        DisplayError(format!(
                            "Failed to parse number: {} in file: {} because: {}",
                            x, file_name, e
                        ))
                    })
                })
                .collect::<Result<_, _>>()?,
        )))
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "non_solver", author = "Martin Quarda <martin@quarda.cz>")]
pub struct Opts {
    matrix: MatrixFromFile,
    rhs: VectorFromFile,
    epsilon: f64,
    method: String,
}
