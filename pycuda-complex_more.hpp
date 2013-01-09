

#ifndef PYCUDA_COMPLEX_HPP_MORE
#define PYCUDA_COMPLEX_HPP_MORE

#include <pycuda/pycuda-complex.hpp>

extern "C++" {
	namespace pycuda{

/*
	__device__ inline complex<double> operator+(const complex<double>& __z, const float& __x)
{return pycuda::complex<double>(__x + __z._M_re, __z._M_im);}


	__device__ inline complex<double> operator+(const complex<float>& __z, const double& __x)
{return complex<double>(__x + __z._M_re, __z._M_im);}
*/

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator+(const complex<_Tp1>& __z, const _Tp2& __x)
	{return complex<double>(__z._M_re + __x, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator+(const _Tp2& __x, const complex<_Tp1>& __z)
	{return complex<double>(__x + __z._M_re, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator-(const complex<_Tp1>& __z, const _Tp2& __x)
	{return complex<double>(__z._M_re - __x, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator-(const _Tp2& __x, const complex<_Tp1>& __z)
	{return complex<double>(__x - __z._M_re, __z._M_im);}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator*(const complex<_Tp1>& __z, const _Tp2& __x)
	{return (complex<double>)__z * __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator*(const _Tp2& __x, const complex<_Tp1>& __z)
	{return (complex<double>)__z * __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator/(const complex<_Tp1>& __z, const _Tp2& __x)
	{return (complex<double>)__z / __x;}

	template <class _Tp1, class _Tp2>
	__device__ inline complex<double> operator/(const _Tp2& __x, const complex<_Tp1>& __z)
	{return __x / (complex<double>)__z;}

	}
}
#endif
