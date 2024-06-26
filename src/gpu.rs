use ark_ec::{CurveGroup, Group, VariableBaseMSM};
use ark_ff::{FftField, PrimeField};
use ark_poly::{
    domain::{radix2::Elements, DomainCoeff},
    EvaluationDomain, Radix2EvaluationDomain,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use delegate::delegate;

#[derive(Copy, Clone, Hash, Eq, PartialEq, CanonicalDeserialize, CanonicalSerialize, Debug)]
pub struct GpuDomain<F: FftField>(Radix2EvaluationDomain<F>, [F; 64]);

trait GpuDomainCoeff<F: FftField>: Sized + DomainCoeff<F> {
    fn fft_in_place(domain: &GpuDomain<F>, input: &mut Vec<Self>);

    fn ifft_in_place(domain: &GpuDomain<F>, input: &mut Vec<Self>);
}

impl<F: FftField, T: DomainCoeff<F>> GpuDomainCoeff<F> for T {
    default fn fft_in_place(domain: &GpuDomain<F>, input: &mut Vec<Self>) {
        let coefficients_time = start_timer!(|| format!("CPU FFT {}", input.len()));
        domain.0.fft_in_place(input);
        end_timer!(coefficients_time);
    }

    default fn ifft_in_place(domain: &GpuDomain<F>, input: &mut Vec<Self>) {
        let coefficients_time = start_timer!(|| format!("CPU iFFT {}", input.len()));
        domain.0.ifft_in_place(input);
        end_timer!(coefficients_time);
    }
}

impl<F: FftField> EvaluationDomain<F> for GpuDomain<F> {
    fn fft_in_place<T: DomainCoeff<F>>(&self, coeffs: &mut Vec<T>) {
        GpuDomainCoeff::fft_in_place(&self, coeffs)
    }

    fn ifft_in_place<T: DomainCoeff<F>>(&self, evals: &mut Vec<T>) {
        GpuDomainCoeff::ifft_in_place(&self, evals)
    }

    type Elements = Elements<F>;

    fn new(num_coeffs: usize) -> Option<Self> {
        Radix2EvaluationDomain::new(num_coeffs).map(Self::from_domain)
    }

    fn get_coset(&self, offset: F) -> Option<Self> {
        self.0.get_coset(offset).map(Self::from_domain)
    }

    fn compute_size_of_domain(num_coeffs: usize) -> Option<usize> {
        Radix2EvaluationDomain::<F>::compute_size_of_domain(num_coeffs)
    }

    delegate! {
        to self.0 {
            fn size_inv(&self) -> F;
            fn size(&self) -> usize;
            fn log_size_of_group(&self) -> u64;
            fn group_gen(&self) -> F;
            fn group_gen_inv(&self) -> F;
            fn coset_offset(&self) -> F;
            fn coset_offset_inv(&self) -> F;
            fn coset_offset_pow_size(&self) -> F;
            fn elements(&self) -> Self::Elements;
        }
    }
}

impl<F: FftField> GpuDomain<F> {
    fn from_domain(domain: Radix2EvaluationDomain<F>) -> Self {
        let mut omega_cache = [F::zero(); 64];
        let omegas = &mut omega_cache[0..32];
        omegas[0] = domain.group_gen;
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }

        let inv_omegas = &mut omega_cache[32..64];
        inv_omegas[0] = domain.group_gen_inv;
        for i in 1..32 {
            inv_omegas[i] = inv_omegas[i - 1].square();
        }
        Self(domain, omega_cache)
    }
}

type FrInt<T> = <<T as Group>::ScalarField as PrimeField>::BigInt;
pub trait GpuVariableBaseMSM: VariableBaseMSM {
    fn msm_bigint_gpu(bases: &[Self::MulBase], bigints: &[FrInt<Self>]) -> Self;
}

impl<T: CurveGroup> GpuVariableBaseMSM for T {
    default fn msm_bigint_gpu(bases: &[Self::MulBase], bigints: &[FrInt<Self>]) -> Self {
        let coefficients_time = start_timer!(|| format!("CPU MSM {}", bases.len()));
        let answer = <Self as VariableBaseMSM>::msm_bigint(bases, bigints);
        end_timer!(coefficients_time);
        answer
    }
}

#[cfg(feature = "cuda")]
mod gpu_impl {
    use super::{FrInt, GpuDomain, GpuDomainCoeff, GpuVariableBaseMSM};

    #[cfg(feature = "cuda-bls12-381")]
    #[allow(unused)]
    use ark_bls12_381::{self, g1::Config as G1Config, g2::Config as G2Config, Fr, G1Projective};
    #[cfg(feature = "cuda-bn254")]
    #[allow(unused)]
    use ark_bn254::{self, g1::Config as G1Config, g2::Config as G2Config, Fr, G1Projective};

    use ark_ec::short_weierstrass::Projective;
    use rayon::prelude::*;

    #[cfg(feature = "g1-fft")]
    impl GpuDomainCoeff<Fr> for G1Projective {
        fn fft_in_place(domain: &GpuDomain<Fr>, input: &mut Vec<Self>) {
            let coefficients_time = start_timer!(|| format!("GPU FFT {}", input.len()));
            ag_cuda_ec::fft::radix_fft_mt(input, &domain.1[0..32], Some(domain.0.offset)).unwrap();
            end_timer!(coefficients_time);
        }

        fn ifft_in_place(domain: &GpuDomain<Fr>, input: &mut Vec<Self>) {
            let coefficients_time = start_timer!(|| format!("GPU iFFT {}", input.len()));
            ag_cuda_ec::fft::radix_ifft_mt(
                input,
                &domain.1[32..64],
                Some(domain.0.offset_inv),
                domain.0.size_inv,
            )
            .unwrap();
            end_timer!(coefficients_time);
        }
    }

    impl GpuDomainCoeff<Fr> for Fr {
        fn fft_in_place(domain: &GpuDomain<Fr>, input: &mut Vec<Self>) {
            let coefficients_time = start_timer!(|| format!("GPU FFT {}", input.len()));
            ag_cuda_ec::fft::radix_fft_mt(input, &domain.1[0..32], Some(domain.0.offset)).unwrap();
            end_timer!(coefficients_time);
        }

        fn ifft_in_place(domain: &GpuDomain<Fr>, input: &mut Vec<Self>) {
            let coefficients_time = start_timer!(|| format!("GPU iFFT {}", input.len()));
            ag_cuda_ec::fft::radix_ifft_mt(
                input,
                &domain.1[32..64],
                Some(domain.0.offset_inv),
                domain.0.size_inv,
            )
            .unwrap();
            end_timer!(coefficients_time);
        }
    }

    impl GpuVariableBaseMSM for Projective<G1Config> {
        fn msm_bigint_gpu(bases: &[Self::MulBase], bigints: &[FrInt<Self>]) -> Self {
            let coefficients_time = start_timer!(|| format!("GPU MSM {}", bases.len()));
            let answer = ag_cuda_ec::multiexp::multiexp_mt(bases, bigints, 256, 7, false)
                .unwrap()
                .par_iter()
                .cloned()
                .sum();
            end_timer!(coefficients_time);
            answer
        }
    }

    impl GpuVariableBaseMSM for Projective<G2Config> {
        fn msm_bigint_gpu(bases: &[Self::MulBase], bigints: &[FrInt<Self>]) -> Self {
            let coefficients_time = start_timer!(|| format!("GPU MSM {}", bases.len()));
            let answer = ag_cuda_ec::multiexp::multiexp_mt(bases, bigints, 256, 7, false)
                .unwrap()
                .par_iter()
                .cloned()
                .sum();
            end_timer!(coefficients_time);
            answer
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use ag_cuda_ec::{pairing_suite::Scalar, test_tools::random_input};
    use ark_ff::{FftField, Field};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use ark_std::{rand::thread_rng, Zero};
    use std::time::Instant;

    use super::*;

    #[cfg(feature = "cuda-bls12-381")]
    use ark_bls12_381::{self, Fr, G1Projective};
    #[cfg(feature = "cuda-bn254")]
    use ark_bn254::{self, Fr, G1Projective};

    #[test]
    fn test_gpu_domain() {
        let mut rng = thread_rng();
        for degree in 4..8 {
            let n = 1 << degree;

            let mut omegas = vec![Scalar::zero(); 32];
            omegas[0] = Scalar::get_root_of_unity(n as u64).unwrap();
            for i in 1..32 {
                omegas[i] = omegas[i - 1].square();
            }
            let original_coeffs: Vec<Fr> = random_input(n, &mut rng);
            let mut fft_coeffs = original_coeffs.clone();

            // Perform FFT
            let domain: GpuDomain<Fr> = EvaluationDomain::new(fft_coeffs.len()).unwrap();
            domain.fft_in_place(&mut fft_coeffs);
            if original_coeffs == fft_coeffs {
                panic!("FFT results do not change");
            }

            // Perform IFFT
            domain.ifft_in_place(&mut fft_coeffs);

            // Compare the IFFT result with the original coefficients
            if original_coeffs != fft_coeffs {
                panic!("FFT and IFFT results do not match the original coefficients");
            }
        }
    }

    #[test]
    fn test_gpu_and_cpu() {
        let mut rng = thread_rng();
        let degree = 15;
        let n = 1 << degree;

        let mut omegas = vec![Scalar::zero(); 32];
        omegas[0] = Scalar::get_root_of_unity(n as u64).unwrap();
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }

        let mut gpu_coeffs: Vec<G1Projective> = random_input(n, &mut rng);
        let mut cpu_coeffs = gpu_coeffs.clone();

        let gpu_domain: GpuDomain<Fr> = EvaluationDomain::new(gpu_coeffs.len()).unwrap();

        // GPU FFT
        let now = Instant::now();
        gpu_domain.fft_in_place(&mut gpu_coeffs);
        let gpu_dur = now.elapsed().as_millis();
        println!("GPU domain took {}ms.", gpu_dur);

        let cpu_domain: Radix2EvaluationDomain<Fr> =
            EvaluationDomain::new(cpu_coeffs.len()).unwrap();
        // CPU FFT
        let now = Instant::now();
        cpu_domain.fft_in_place(&mut cpu_coeffs);
        let gpu_dur = now.elapsed().as_millis();
        println!("CPU domain took {}ms.", gpu_dur);
    }
}
