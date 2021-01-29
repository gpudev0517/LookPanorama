#include "3DMath.h"
#include <QEasingCurve>


#ifndef _WIN32
#define _isnan isnan
#define _finite finite
#endif

sd_scalar vec2::normalize()
{
	sd_scalar norm = sqrtf(x * x + y * y);
	if (norm > sd_eps)
		norm = sd_one / norm;
	else
		norm = sd_zero;
	x *= norm;
	y *= norm;
	return norm;
}

mat3::mat3()
{
}

mat3::mat3(const sd_scalar* array)
{
	memcpy(mat_array, array, sizeof(sd_scalar) * 9);
}

mat3::mat3(const mat3 & M)
{
	memcpy(mat_array, M.mat_array, sizeof(sd_scalar) * 9);
}

mat4::mat4()
{
}

mat4::mat4(const sd_scalar* array)
{
	memcpy(mat_array, array, sizeof(sd_scalar) * 16);
}

mat4::mat4(const mat4& M)
{
	memcpy(mat_array, M.mat_array, sizeof(sd_scalar) * 16);
}

vec3 & cross(vec3 & u, const vec3 & v, const vec3 & w)
{
	vec3 temp;
	temp.x = v.y*w.z - v.z*w.y;
	temp.y = v.z*w.x - v.x*w.z;
	temp.z = v.x*w.y - v.y*w.x;
	u = temp;
	return u;
}


sd_scalar & dot(sd_scalar& u, const vec3& v, const vec3& w)
{
	u = v.x*w.x + v.y*w.y + v.z*w.z;
	return u;
}

sd_scalar dot(const vec3& v, const vec3& w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z;
}

sd_scalar & dot(sd_scalar& u, const vec4& v, const vec4& w)
{
	u = v.x*w.x + v.y*w.y + v.z*w.z + v.w*w.w;
	return u;
}

sd_scalar dot(const vec4& v, const vec4& w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z + v.w*w.w;
}

sd_scalar & dot(sd_scalar& u, const vec3& v, const vec4& w)
{
	u = v.x*w.x + v.y*w.y + v.z*w.z;
	return u;
}

sd_scalar dot(const vec3& v, const vec4& w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z;
}

sd_scalar & dot(sd_scalar& u, const vec4& v, const vec3& w)
{
	u = v.x*w.x + v.y*w.y + v.z*w.z;
	return u;
}

sd_scalar dot(const vec4& v, const vec3& w)
{
	return v.x*w.x + v.y*w.y + v.z*w.z;
}

vec3 & reflect(vec3& r, const vec3& n, const vec3& l)
{
	sd_scalar n_dot_l;
	n_dot_l = sd_two * dot(n_dot_l,n,l);
	mult(r,l,-sd_one);
	madd(r,n,n_dot_l);
	return r;
}

vec3 & madd(vec3 & u, const vec3& v, const sd_scalar& lambda)
{
	u.x += v.x*lambda;
	u.y += v.y*lambda;
	u.z += v.z*lambda;
	return u;
}

vec3 & mult(vec3 & u, const vec3& v, const sd_scalar& lambda)
{
	u.x = v.x*lambda;
	u.y = v.y*lambda;
	u.z = v.z*lambda;
	return u;
}

vec3 & mult(vec3 & u, const vec3& v, const vec3& w)
{
	u.x = v.x*w.x;
	u.y = v.y*w.y;
	u.z = v.z*w.z;
	return u;
}

vec3 & sub(vec3 & u, const vec3& v, const vec3& w)
{
	u.x = v.x - w.x;
	u.y = v.y - w.y;
	u.z = v.z - w.z;
	return u;
}

vec3 & add(vec3 & u, const vec3& v, const vec3& w)
{
	u.x = v.x + w.x;
	u.y = v.y + w.y;
	u.z = v.z + w.z;
	return u;
}

void vec3::orthogonalize( const vec3& v )
{
	//  determine the orthogonal projection of this on v : dot( v , this ) * v
	//  and subtract it from this resulting in the orthogonalized this
	vec3 res = dot( v, vec3(x, y, z) ) * v;
	x -= res.x;
	y -= res.y;
	z -= res.y;
}

sd_scalar vec3::normalize()
{
	sd_scalar norm = sqrtf(x * x + y * y + z * z);
	if (norm > sd_eps)
		norm = sd_one / norm;
	else
		norm = sd_zero;
	x *= norm;
	y *= norm;
	z *= norm;
	return norm;
}

vec2 & scale(vec2& u, const sd_scalar s)
{
	u.x *= s;
	u.y *= s;
	return u;
}

vec3 & scale(vec3& u, const sd_scalar s)
{
	u.x *= s;
	u.y *= s;
	u.z *= s;
	return u;
}

vec4 & scale(vec4& u, const sd_scalar s)
{
	u.x *= s;
	u.y *= s;
	u.z *= s;
	u.w *= s;
	return u;
}

vec3 & mult(vec3& u, const mat3& M, const vec3& v)
{
	u.x = M.a00 * v.x + M.a01 * v.y + M.a02 * v.z;
	u.y = M.a10 * v.x + M.a11 * v.y + M.a12 * v.z;
	u.z = M.a20 * v.x + M.a21 * v.y + M.a22 * v.z;
	return u;
}

vec3 & mult(vec3& u, const vec3& v, const mat3& M)
{
	u.x = M.a00 * v.x + M.a10 * v.y + M.a20 * v.z;
	u.y = M.a01 * v.x + M.a11 * v.y + M.a21 * v.z;
	u.z = M.a02 * v.x + M.a12 * v.y + M.a22 * v.z;
	return u;
}

const vec3 operator*(const mat3& M, const vec3& v)
{
	vec3 u;
	u.x = M.a00 * v.x + M.a01 * v.y + M.a02 * v.z;
	u.y = M.a10 * v.x + M.a11 * v.y + M.a12 * v.z;
	u.z = M.a20 * v.x + M.a21 * v.y + M.a22 * v.z;
	return u;
}

const vec3 operator*(const vec3& v, const mat3& M)
{
	vec3 u;
	u.x = M.a00 * v.x + M.a10 * v.y + M.a20 * v.z;
	u.y = M.a01 * v.x + M.a11 * v.y + M.a21 * v.z;
	u.z = M.a02 * v.x + M.a12 * v.y + M.a22 * v.z;
	return u;
}

vec4 & mult(vec4& u, const mat4& M, const vec4& v)
{
	u.x = M.a00 * v.x + M.a01 * v.y + M.a02 * v.z + M.a03 * v.w;
	u.y = M.a10 * v.x + M.a11 * v.y + M.a12 * v.z + M.a13 * v.w;
	u.z = M.a20 * v.x + M.a21 * v.y + M.a22 * v.z + M.a23 * v.w;
	u.w = M.a30 * v.x + M.a31 * v.y + M.a32 * v.z + M.a33 * v.w;
	return u;
}

vec4 & mult(vec4& u, const vec4& v, const mat4& M)
{
	u.x = M.a00 * v.x + M.a10 * v.y + M.a20 * v.z + M.a30 * v.w;
	u.y = M.a01 * v.x + M.a11 * v.y + M.a21 * v.z + M.a31 * v.w;
	u.z = M.a02 * v.x + M.a12 * v.y + M.a22 * v.z + M.a32 * v.w;
	u.w = M.a03 * v.x + M.a13 * v.y + M.a23 * v.z + M.a33 * v.w;
	return u;
}

const vec4 operator*(const mat4& M, const vec4& v)
{
	vec4 u;
	u.x = M.a00 * v.x + M.a01 * v.y + M.a02 * v.z + M.a03 * v.w;
	u.y = M.a10 * v.x + M.a11 * v.y + M.a12 * v.z + M.a13 * v.w;
	u.z = M.a20 * v.x + M.a21 * v.y + M.a22 * v.z + M.a23 * v.w;
	u.w = M.a30 * v.x + M.a31 * v.y + M.a32 * v.z + M.a33 * v.w;
	return u;
}

const vec4 operator*(const vec4& v, const mat4& M)
{
	vec4 u;
	u.x = M.a00 * v.x + M.a10 * v.y + M.a20 * v.z + M.a30 * v.w;
	u.y = M.a01 * v.x + M.a11 * v.y + M.a21 * v.z + M.a31 * v.w;
	u.z = M.a02 * v.x + M.a12 * v.y + M.a22 * v.z + M.a32 * v.w;
	u.w = M.a03 * v.x + M.a13 * v.y + M.a23 * v.z + M.a33 * v.w;
	return u;
}

vec3 & mult_pos(vec3& u, const mat4& M, const vec3& v)
{
	sd_scalar oow;
	sd_scalar divider = v.x * M.a30 + v.y * M.a31 + v.z * M.a32 + M.a33;
	if (divider < sd_eps && divider > -sd_eps)
		oow = sd_one ;
	else
		oow = sd_one / divider;
	u.x = (M.a00 * v.x + M.a01 * v.y + M.a02 * v.z + M.a03) * oow;
	u.y = (M.a10 * v.x + M.a11 * v.y + M.a12 * v.z + M.a13) * oow;
	u.z = (M.a20 * v.x + M.a21 * v.y + M.a22 * v.z + M.a23) * oow;
	return u;
}

vec3 & mult_pos(vec3& u, const vec3& v, const mat4& M)
{
	sd_scalar oow;
	sd_scalar divider = v.x * M.a03 + v.y * M.a13 + v.z * M.a23 + M.a33;
	if (divider < sd_eps && divider > -sd_eps)
		oow = sd_one ;
	else
		oow = sd_one / divider;

	u.x = (M.a00 * v.x + M.a10 * v.y + M.a20 * v.z + M.a30) * oow;
	u.y = (M.a01 * v.x + M.a11 * v.y + M.a21 * v.z + M.a31) * oow;
	u.z = (M.a02 * v.x + M.a12 * v.y + M.a22 * v.z + M.a32) * oow;
	return u;
}

vec3 & mult_dir(vec3& u, const mat4& M, const vec3& v)
{
	u.x = M.a00 * v.x + M.a01 * v.y + M.a02 * v.z;
	u.y = M.a10 * v.x + M.a11 * v.y + M.a12 * v.z;
	u.z = M.a20 * v.x + M.a21 * v.y + M.a22 * v.z;
	return u;
}

vec3 & mult_dir(vec3& u, const vec3& v, const mat4& M)
{
	u.x = M.a00 * v.x + M.a10 * v.y + M.a20 * v.z;
	u.y = M.a01 * v.x + M.a11 * v.y + M.a21 * v.z;
	u.z = M.a02 * v.x + M.a12 * v.y + M.a22 * v.z;
	return u;
}

vec3 & mult(vec3& u, const mat4& M, const vec3& v)
{
	u.x = M.a00 * v.x + M.a01 * v.y + M.a02 * v.z + M.a03;
	u.y = M.a10 * v.x + M.a11 * v.y + M.a12 * v.z + M.a13;
	u.z = M.a20 * v.x + M.a21 * v.y + M.a22 * v.z + M.a23;
	return u;
}

vec3 & mult(vec3& u, const vec3& v, const mat4& M)
{
	u.x = M.a00 * v.x + M.a10 * v.y + M.a20 * v.z + M.a30;
	u.y = M.a01 * v.x + M.a11 * v.y + M.a21 * v.z + M.a31;
	u.z = M.a02 * v.x + M.a12 * v.y + M.a22 * v.z + M.a32;
	return u;
}

mat4 & add(mat4& A, const mat4& B)
{
	A.a00 += B.a00;
	A.a10 += B.a10;
	A.a20 += B.a20;
	A.a30 += B.a30;
	A.a01 += B.a01;
	A.a11 += B.a11;
	A.a21 += B.a21;
	A.a31 += B.a31;
	A.a02 += B.a02;
	A.a12 += B.a12;
	A.a22 += B.a22;
	A.a32 += B.a32;
	A.a03 += B.a03;
	A.a13 += B.a13;
	A.a23 += B.a23;
	A.a33 += B.a33;
	return A;
}

mat3 & add(mat3& A, const mat3& B)
{
	A.a00 += B.a00;
	A.a10 += B.a10;
	A.a20 += B.a20;
	A.a01 += B.a01;
	A.a11 += B.a11;
	A.a21 += B.a21;
	A.a02 += B.a02;
	A.a12 += B.a12;
	A.a22 += B.a22;
	return A;
}


// Computes C = A + B
mat4 & add(mat4 & C, const mat4 & A, const mat4 & B)
{
	// If there is selfassignment involved
	// we can't go without a temporary.
	if (&C == &A || &C == &B)
	{
		mat4 mTemp;

		mTemp.a00 = A.a00 + B.a00;
		mTemp.a01 = A.a01 + B.a01;
		mTemp.a02 = A.a02 + B.a02;
		mTemp.a03 = A.a03 + B.a03;
		mTemp.a10 = A.a10 + B.a10;
		mTemp.a11 = A.a11 + B.a11;
		mTemp.a12 = A.a12 + B.a12;
		mTemp.a13 = A.a13 + B.a13;
		mTemp.a20 = A.a20 + B.a20;
		mTemp.a21 = A.a21 + B.a21;
		mTemp.a22 = A.a22 + B.a22;
		mTemp.a23 = A.a23 + B.a23;
		mTemp.a30 = A.a30 + B.a30;
		mTemp.a31 = A.a31 + B.a31;
		mTemp.a32 = A.a32 + B.a32;
		mTemp.a33 = A.a33 + B.a33;

		C = mTemp;
	}
	else
	{
		C.a00 = A.a00 + B.a00;
		C.a01 = A.a01 + B.a01;
		C.a02 = A.a02 + B.a02;
		C.a03 = A.a03 + B.a03;
		C.a10 = A.a10 + B.a10;
		C.a11 = A.a11 + B.a11;
		C.a12 = A.a12 + B.a12;
		C.a13 = A.a13 + B.a13;
		C.a20 = A.a20 + B.a20;
		C.a21 = A.a21 + B.a21;
		C.a22 = A.a22 + B.a22;
		C.a23 = A.a23 + B.a23;
		C.a30 = A.a30 + B.a30;
		C.a31 = A.a31 + B.a31;
		C.a32 = A.a32 + B.a32;
		C.a33 = A.a33 + B.a33;
	}
	return C;
}

mat3 & add(mat3 & C, const mat3 & A, const mat3 & B)
{
	// If there is selfassignment involved
	// we can't go without a temporary.
	if (&C == &A || &C == &B)
	{
		mat3 mTemp;

		mTemp.a00 = A.a00 + B.a00;
		mTemp.a01 = A.a01 + B.a01;
		mTemp.a02 = A.a02 + B.a02;
		mTemp.a10 = A.a10 + B.a10;
		mTemp.a11 = A.a11 + B.a11;
		mTemp.a12 = A.a12 + B.a12;
		mTemp.a20 = A.a20 + B.a20;
		mTemp.a21 = A.a21 + B.a21;
		mTemp.a22 = A.a22 + B.a22;

		C = mTemp;
	}
	else
	{
		C.a00 = A.a00 + B.a00;
		C.a01 = A.a01 + B.a01;
		C.a02 = A.a02 + B.a02;
		C.a10 = A.a10 + B.a10;
		C.a11 = A.a11 + B.a11;
		C.a12 = A.a12 + B.a12;
		C.a20 = A.a20 + B.a20;
		C.a21 = A.a21 + B.a21;
		C.a22 = A.a22 + B.a22;
	}
	return C;
}


// C = A * B

// C.a00 C.a01 C.a02 C.a03   A.a00 A.a01 A.a02 A.a03   B.a00 B.a01 B.a02 B.a03
//                                                                            
// C.a10 C.a11 C.a12 C.a13   A.a10 A.a11 A.a12 A.a13   B.a10 B.a11 B.a12 B.a13
//                                                                         
// C.a20 C.a21 C.a22 C.a23   A.a20 A.a21 A.a22 A.a23   B.a20 B.a21 B.a22 B.a23  
//                                                                            
// C.a30 C.a31 C.a32 C.a33 = A.a30 A.a31 A.a32 A.a33 * B.a30 B.a31 B.a32 B.a33

mat4 & mult(mat4& C, const mat4& A, const mat4& B)
{
	// If there is selfassignment involved
	// we can't go without a temporary.
	if (&C == &A || &C == &B)
	{
		mat4 mTemp;

		mTemp.a00 = A.a00 * B.a00 + A.a01 * B.a10 + A.a02 * B.a20 + A.a03 * B.a30;
		mTemp.a10 = A.a10 * B.a00 + A.a11 * B.a10 + A.a12 * B.a20 + A.a13 * B.a30;
		mTemp.a20 = A.a20 * B.a00 + A.a21 * B.a10 + A.a22 * B.a20 + A.a23 * B.a30;
		mTemp.a30 = A.a30 * B.a00 + A.a31 * B.a10 + A.a32 * B.a20 + A.a33 * B.a30;
		mTemp.a01 = A.a00 * B.a01 + A.a01 * B.a11 + A.a02 * B.a21 + A.a03 * B.a31;
		mTemp.a11 = A.a10 * B.a01 + A.a11 * B.a11 + A.a12 * B.a21 + A.a13 * B.a31;
		mTemp.a21 = A.a20 * B.a01 + A.a21 * B.a11 + A.a22 * B.a21 + A.a23 * B.a31;
		mTemp.a31 = A.a30 * B.a01 + A.a31 * B.a11 + A.a32 * B.a21 + A.a33 * B.a31;
		mTemp.a02 = A.a00 * B.a02 + A.a01 * B.a12 + A.a02 * B.a22 + A.a03 * B.a32;
		mTemp.a12 = A.a10 * B.a02 + A.a11 * B.a12 + A.a12 * B.a22 + A.a13 * B.a32;
		mTemp.a22 = A.a20 * B.a02 + A.a21 * B.a12 + A.a22 * B.a22 + A.a23 * B.a32;
		mTemp.a32 = A.a30 * B.a02 + A.a31 * B.a12 + A.a32 * B.a22 + A.a33 * B.a32;
		mTemp.a03 = A.a00 * B.a03 + A.a01 * B.a13 + A.a02 * B.a23 + A.a03 * B.a33;
		mTemp.a13 = A.a10 * B.a03 + A.a11 * B.a13 + A.a12 * B.a23 + A.a13 * B.a33;
		mTemp.a23 = A.a20 * B.a03 + A.a21 * B.a13 + A.a22 * B.a23 + A.a23 * B.a33;
		mTemp.a33 = A.a30 * B.a03 + A.a31 * B.a13 + A.a32 * B.a23 + A.a33 * B.a33;

		C = mTemp;
	}
	else
	{
		C.a00 = A.a00 * B.a00 + A.a01 * B.a10 + A.a02 * B.a20 + A.a03 * B.a30;
		C.a10 = A.a10 * B.a00 + A.a11 * B.a10 + A.a12 * B.a20 + A.a13 * B.a30;
		C.a20 = A.a20 * B.a00 + A.a21 * B.a10 + A.a22 * B.a20 + A.a23 * B.a30;
		C.a30 = A.a30 * B.a00 + A.a31 * B.a10 + A.a32 * B.a20 + A.a33 * B.a30;
		C.a01 = A.a00 * B.a01 + A.a01 * B.a11 + A.a02 * B.a21 + A.a03 * B.a31;
		C.a11 = A.a10 * B.a01 + A.a11 * B.a11 + A.a12 * B.a21 + A.a13 * B.a31;
		C.a21 = A.a20 * B.a01 + A.a21 * B.a11 + A.a22 * B.a21 + A.a23 * B.a31;
		C.a31 = A.a30 * B.a01 + A.a31 * B.a11 + A.a32 * B.a21 + A.a33 * B.a31;
		C.a02 = A.a00 * B.a02 + A.a01 * B.a12 + A.a02 * B.a22 + A.a03 * B.a32;
		C.a12 = A.a10 * B.a02 + A.a11 * B.a12 + A.a12 * B.a22 + A.a13 * B.a32;
		C.a22 = A.a20 * B.a02 + A.a21 * B.a12 + A.a22 * B.a22 + A.a23 * B.a32;
		C.a32 = A.a30 * B.a02 + A.a31 * B.a12 + A.a32 * B.a22 + A.a33 * B.a32;
		C.a03 = A.a00 * B.a03 + A.a01 * B.a13 + A.a02 * B.a23 + A.a03 * B.a33;
		C.a13 = A.a10 * B.a03 + A.a11 * B.a13 + A.a12 * B.a23 + A.a13 * B.a33;
		C.a23 = A.a20 * B.a03 + A.a21 * B.a13 + A.a22 * B.a23 + A.a23 * B.a33;
		C.a33 = A.a30 * B.a03 + A.a31 * B.a13 + A.a32 * B.a23 + A.a33 * B.a33;
	}

	return C;
}

mat4 mat4::operator*(const mat4& B) const
{
	mat4 C;
	C.a00 = a00 * B.a00 + a01 * B.a10 + a02 * B.a20 + a03 * B.a30;
	C.a10 = a10 * B.a00 + a11 * B.a10 + a12 * B.a20 + a13 * B.a30;
	C.a20 = a20 * B.a00 + a21 * B.a10 + a22 * B.a20 + a23 * B.a30;
	C.a30 = a30 * B.a00 + a31 * B.a10 + a32 * B.a20 + a33 * B.a30;
	C.a01 = a00 * B.a01 + a01 * B.a11 + a02 * B.a21 + a03 * B.a31;
	C.a11 = a10 * B.a01 + a11 * B.a11 + a12 * B.a21 + a13 * B.a31;
	C.a21 = a20 * B.a01 + a21 * B.a11 + a22 * B.a21 + a23 * B.a31;
	C.a31 = a30 * B.a01 + a31 * B.a11 + a32 * B.a21 + a33 * B.a31;
	C.a02 = a00 * B.a02 + a01 * B.a12 + a02 * B.a22 + a03 * B.a32;
	C.a12 = a10 * B.a02 + a11 * B.a12 + a12 * B.a22 + a13 * B.a32;
	C.a22 = a20 * B.a02 + a21 * B.a12 + a22 * B.a22 + a23 * B.a32;
	C.a32 = a30 * B.a02 + a31 * B.a12 + a32 * B.a22 + a33 * B.a32;
	C.a03 = a00 * B.a03 + a01 * B.a13 + a02 * B.a23 + a03 * B.a33;
	C.a13 = a10 * B.a03 + a11 * B.a13 + a12 * B.a23 + a13 * B.a33;
	C.a23 = a20 * B.a03 + a21 * B.a13 + a22 * B.a23 + a23 * B.a33;
	C.a33 = a30 * B.a03 + a31 * B.a13 + a32 * B.a23 + a33 * B.a33;
	return C;
}

// C = A * B

// C.a00 C.a01 C.a02   A.a00 A.a01 A.a02   B.a00 B.a01 B.a02
//                                                          
// C.a10 C.a11 C.a12   A.a10 A.a11 A.a12   B.a10 B.a11 B.a12
//                                                          
// C.a20 C.a21 C.a22 = A.a20 A.a21 A.a22 * B.a20 B.a21 B.a22

mat3 & mult(mat3& C, const mat3& A, const mat3& B)
{
	// If there is sel fassignment involved
	// we can't go without a temporary.
	if (&C == &A || &C == &B)
	{
		mat3 mTemp;

		mTemp.a00 = A.a00 * B.a00 + A.a01 * B.a10 + A.a02 * B.a20;
		mTemp.a10 = A.a10 * B.a00 + A.a11 * B.a10 + A.a12 * B.a20;
		mTemp.a20 = A.a20 * B.a00 + A.a21 * B.a10 + A.a22 * B.a20;
		mTemp.a01 = A.a00 * B.a01 + A.a01 * B.a11 + A.a02 * B.a21;
		mTemp.a11 = A.a10 * B.a01 + A.a11 * B.a11 + A.a12 * B.a21;
		mTemp.a21 = A.a20 * B.a01 + A.a21 * B.a11 + A.a22 * B.a21;
		mTemp.a02 = A.a00 * B.a02 + A.a01 * B.a12 + A.a02 * B.a22;
		mTemp.a12 = A.a10 * B.a02 + A.a11 * B.a12 + A.a12 * B.a22;
		mTemp.a22 = A.a20 * B.a02 + A.a21 * B.a12 + A.a22 * B.a22;

		C = mTemp;
	}
	else
	{
		C.a00 = A.a00 * B.a00 + A.a01 * B.a10 + A.a02 * B.a20;
		C.a10 = A.a10 * B.a00 + A.a11 * B.a10 + A.a12 * B.a20;
		C.a20 = A.a20 * B.a00 + A.a21 * B.a10 + A.a22 * B.a20;
		C.a01 = A.a00 * B.a01 + A.a01 * B.a11 + A.a02 * B.a21;
		C.a11 = A.a10 * B.a01 + A.a11 * B.a11 + A.a12 * B.a21;
		C.a21 = A.a20 * B.a01 + A.a21 * B.a11 + A.a22 * B.a21;
		C.a02 = A.a00 * B.a02 + A.a01 * B.a12 + A.a02 * B.a22;
		C.a12 = A.a10 * B.a02 + A.a11 * B.a12 + A.a12 * B.a22;
		C.a22 = A.a20 * B.a02 + A.a21 * B.a12 + A.a22 * B.a22;
	}

	return C;
}

mat3 mult(const mat3 & A, const mat3 & B)
{
	mat3 _result;
	mult(_result, A, B);
	return _result;
}


mat3 & transpose(mat3& A)
{
	sd_scalar tmp;
	tmp = A.a01;
	A.a01 = A.a10;
	A.a10 = tmp;

	tmp = A.a02;
	A.a02 = A.a20;
	A.a20 = tmp;

	tmp = A.a12;
	A.a12 = A.a21;
	A.a21 = tmp;
	return A;
}

mat4 & transpose(mat4& A)
{
	sd_scalar tmp;
	tmp = A.a01;
	A.a01 = A.a10;
	A.a10 = tmp;

	tmp = A.a02;
	A.a02 = A.a20;
	A.a20 = tmp;

	tmp = A.a03;
	A.a03 = A.a30;
	A.a30 = tmp;

	tmp = A.a12;
	A.a12 = A.a21;
	A.a21 = tmp;

	tmp = A.a13;
	A.a13 = A.a31;
	A.a31 = tmp;

	tmp = A.a23;
	A.a23 = A.a32;
	A.a32 = tmp;
	return A;
}

mat4 & transpose(mat4& B, const mat4& A)
{
	B.a00 = A.a00;
	B.a01 = A.a10;
	B.a02 = A.a20;
	B.a03 = A.a30;
	B.a10 = A.a01;
	B.a11 = A.a11;
	B.a12 = A.a21;
	B.a13 = A.a31;
	B.a20 = A.a02;
	B.a21 = A.a12;
	B.a22 = A.a22;
	B.a23 = A.a32;
	B.a30 = A.a03;
	B.a31 = A.a13;
	B.a32 = A.a23;
	B.a33 = A.a33;
	return B;
}

mat3 & transpose(mat3& B, const mat3& A)
{
	B.a00 = A.a00;
	B.a01 = A.a10;
	B.a02 = A.a20;
	B.a10 = A.a01;
	B.a11 = A.a11;
	B.a12 = A.a21;
	B.a20 = A.a02;
	B.a21 = A.a12;
	B.a22 = A.a22;
	return B;
}

/*
calculate the determinent of a 2x2 matrix in the from

| a1 a2 |
| b1 b2 |

*/
sd_scalar det2x2(sd_scalar a1, sd_scalar a2, sd_scalar b1, sd_scalar b2)
{
	return a1 * b2 - b1 * a2;
}

/*
calculate the determinent of a 3x3 matrix in the from

| a1 a2 a3 |
| b1 b2 b3 |
| c1 c2 c3 |

*/
sd_scalar det3x3(sd_scalar a1, sd_scalar a2, sd_scalar a3, 
				 sd_scalar b1, sd_scalar b2, sd_scalar b3, 
				 sd_scalar c1, sd_scalar c2, sd_scalar c3)
{
	return a1 * det2x2(b2, b3, c2, c3) - b1 * det2x2(a2, a3, c2, c3) + c1 * det2x2(a2, a3, b2, b3);
}

mat4 & invert(mat4& B, const mat4& A)
{
	sd_scalar det,oodet;

	B.a00 =  det3x3(A.a11, A.a21, A.a31, A.a12, A.a22, A.a32, A.a13, A.a23, A.a33);
	B.a10 = -det3x3(A.a10, A.a20, A.a30, A.a12, A.a22, A.a32, A.a13, A.a23, A.a33);
	B.a20 =  det3x3(A.a10, A.a20, A.a30, A.a11, A.a21, A.a31, A.a13, A.a23, A.a33);
	B.a30 = -det3x3(A.a10, A.a20, A.a30, A.a11, A.a21, A.a31, A.a12, A.a22, A.a32);

	B.a01 = -det3x3(A.a01, A.a21, A.a31, A.a02, A.a22, A.a32, A.a03, A.a23, A.a33);
	B.a11 =  det3x3(A.a00, A.a20, A.a30, A.a02, A.a22, A.a32, A.a03, A.a23, A.a33);
	B.a21 = -det3x3(A.a00, A.a20, A.a30, A.a01, A.a21, A.a31, A.a03, A.a23, A.a33);
	B.a31 =  det3x3(A.a00, A.a20, A.a30, A.a01, A.a21, A.a31, A.a02, A.a22, A.a32);

	B.a02 =  det3x3(A.a01, A.a11, A.a31, A.a02, A.a12, A.a32, A.a03, A.a13, A.a33);
	B.a12 = -det3x3(A.a00, A.a10, A.a30, A.a02, A.a12, A.a32, A.a03, A.a13, A.a33);
	B.a22 =  det3x3(A.a00, A.a10, A.a30, A.a01, A.a11, A.a31, A.a03, A.a13, A.a33);
	B.a32 = -det3x3(A.a00, A.a10, A.a30, A.a01, A.a11, A.a31, A.a02, A.a12, A.a32);

	B.a03 = -det3x3(A.a01, A.a11, A.a21, A.a02, A.a12, A.a22, A.a03, A.a13, A.a23);
	B.a13 =  det3x3(A.a00, A.a10, A.a20, A.a02, A.a12, A.a22, A.a03, A.a13, A.a23);
	B.a23 = -det3x3(A.a00, A.a10, A.a20, A.a01, A.a11, A.a21, A.a03, A.a13, A.a23);
	B.a33 =  det3x3(A.a00, A.a10, A.a20, A.a01, A.a11, A.a21, A.a02, A.a12, A.a22);

	det = (A.a00 * B.a00) + (A.a01 * B.a10) + (A.a02 * B.a20) + (A.a03 * B.a30);

	// The following divions goes unchecked for division
	// by zero. We should consider throwing an exception
	// if det < eps.
	oodet = sd_one / det;

	B.a00 *= oodet;
	B.a10 *= oodet;
	B.a20 *= oodet;
	B.a30 *= oodet;

	B.a01 *= oodet;
	B.a11 *= oodet;
	B.a21 *= oodet;
	B.a31 *= oodet;

	B.a02 *= oodet;
	B.a12 *= oodet;
	B.a22 *= oodet;
	B.a32 *= oodet;

	B.a03 *= oodet;
	B.a13 *= oodet;
	B.a23 *= oodet;
	B.a33 *= oodet;

	return B;
}

mat4 & invert_rot_trans(mat4& B, const mat4& A)
{
	B.a00 = A.a00;
	B.a10 = A.a01;
	B.a20 = A.a02;
	B.a30 = A.a30;
	B.a01 = A.a10;
	B.a11 = A.a11;
	B.a21 = A.a12;
	B.a31 = A.a31;
	B.a02 = A.a20;
	B.a12 = A.a21;
	B.a22 = A.a22;
	B.a32 = A.a32;
	B.a03 = - (A.a00 * A.a03 + A.a10 * A.a13 + A.a20 * A.a23);
	B.a13 = - (A.a01 * A.a03 + A.a11 * A.a13 + A.a21 * A.a23);
	B.a23 = - (A.a02 * A.a03 + A.a12 * A.a13 + A.a22 * A.a23);
	B.a33 = A.a33;
	return B;
}

sd_scalar det(const mat3& A)
{
	return det3x3(A.a00, A.a01, A.a02, 
		A.a10, A.a11, A.a12, 
		A.a20, A.a21, A.a22);
}

mat3 & invert(mat3& B, const mat3& A)
{
	sd_scalar det,oodet;

	B.a00 =  (A.a11 * A.a22 - A.a21 * A.a12);
	B.a10 = -(A.a10 * A.a22 - A.a20 * A.a12);
	B.a20 =  (A.a10 * A.a21 - A.a20 * A.a11);
	B.a01 = -(A.a01 * A.a22 - A.a21 * A.a02);
	B.a11 =  (A.a00 * A.a22 - A.a20 * A.a02);
	B.a21 = -(A.a00 * A.a21 - A.a20 * A.a01);
	B.a02 =  (A.a01 * A.a12 - A.a11 * A.a02);
	B.a12 = -(A.a00 * A.a12 - A.a10 * A.a02);
	B.a22 =  (A.a00 * A.a11 - A.a10 * A.a01);

	det = (A.a00 * B.a00) + (A.a01 * B.a10) + (A.a02 * B.a20);

	oodet = sd_one / det;

	B.a00 *= oodet; B.a01 *= oodet; B.a02 *= oodet;
	B.a10 *= oodet; B.a11 *= oodet; B.a12 *= oodet;
	B.a20 *= oodet; B.a21 *= oodet; B.a22 *= oodet;
	return B;
}

vec2 & normalize(vec2& u)
{
	sd_scalar norm = sqrtf(u.x * u.x + u.y * u.y);
	if (norm > sd_eps)
		norm = sd_one / norm;
	else
		norm = sd_zero;
	return scale(u,norm); 
}

vec3 & normalize(vec3& u)
{
	sd_scalar norm = sqrtf(u.x * u.x + u.y * u.y + u.z * u.z);
	if (norm > sd_eps)
		norm = sd_one / norm;
	else
		norm = sd_zero;
	return scale(u,norm); 
}

vec4 & normalize(vec4& u)
{
	sd_scalar norm = sqrtf(u.x * u.x + u.y * u.y + u.z * u.z + u.w * u.w);
	if (norm > sd_eps)
		norm = sd_one / norm;
	else
		norm = sd_zero;
	return scale(u,norm); 
}

quat & normalize(quat & p)
{
	sd_scalar norm = sqrtf(p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w);
	if (norm > sd_eps)
		norm = sd_one / norm;
	else
		norm = sd_zero;
	p.x *= norm;
	p.y *= norm;
	p.z *= norm;
	p.w *= norm;
	return p; 
}

mat4 & look_at(mat4& M, const vec3& eye, const vec3& center, const vec3& up)
{
	vec3 x, y, z;

	// make rotation matrix

	// Z vector
	z.x = eye.x - center.x;
	z.y = eye.y - center.y;
	z.z = eye.z - center.z;
	normalize(z);

	// Y vector
	y.x = up.x;
	y.y = up.y;
	y.z = up.z;

	// X vector = Y cross Z
	cross(x,y,z);

	// Recompute Y = Z cross X
	cross(y,z,x);

	// cross product gives area of parallelogram, which is < 1.0 for
	// non-perpendicular unit-length vectors; so normalize x, y here
	normalize(x);
	normalize(y);

	M.a00 = x.x; M.a01 = x.y; M.a02 = x.z; M.a03 = -x.x * eye.x - x.y * eye.y - x.z*eye.z;
	M.a10 = y.x; M.a11 = y.y; M.a12 = y.z; M.a13 = -y.x * eye.x - y.y * eye.y - y.z*eye.z;
	M.a20 = z.x; M.a21 = z.y; M.a22 = z.z; M.a23 = -z.x * eye.x - z.y * eye.y - z.z*eye.z;
	M.a30 = sd_zero; M.a31 = sd_zero; M.a32 = sd_zero; M.a33 = sd_one;
	return M;
}

mat4 & frustum(mat4& M, const sd_scalar l, const sd_scalar r, const sd_scalar b, 
			   const sd_scalar t, const sd_scalar n, const sd_scalar f)
{
	M.a00 = (sd_two*n) / (r-l);
	M.a10 = 0.0;
	M.a20 = 0.0;
	M.a30 = 0.0;

	M.a01 = 0.0;
	M.a11 = (sd_two*n) / (t-b);
	M.a21 = 0.0;
	M.a31 = 0.0;

	M.a02 = (r+l) / (r-l);
	M.a12 = (t+b) / (t-b);
	M.a22 = -(f+n) / (f-n);
	M.a32 = -sd_one;

	M.a03 = 0.0;
	M.a13 = 0.0;
	M.a23 = -(sd_two*f*n) / (f-n);
	M.a33 = 0.0;
	return M;
}

mat4 & perspective(mat4& M, const sd_scalar fovy, const sd_scalar aspect, const sd_scalar n, const sd_scalar f)
{
	sd_scalar xmin, xmax, ymin, ymax;

	ymax = n * tanf(fovy * sd_to_rad * sd_zero_5);
	ymin = -ymax;

	xmin = ymin * aspect;
	xmax = ymax * aspect;

	return frustum(M, xmin, xmax, ymin, ymax, n, f);
}

extern mat4 & ortho(mat4 & M, const sd_scalar left, 
					const sd_scalar right, 
					const sd_scalar bottom, 
					const sd_scalar top,
					const sd_scalar n,
					const sd_scalar f)
{
	M.a00 = sd_two / (right - left);
	M.a01 = sd_zero;
	M.a02 = sd_zero;
	M.a03 = - (right + left) / (right - left);
	M.a10 = sd_zero;
	M.a11 = sd_two / (top - bottom);
	M.a12 = sd_zero;
	M.a13 = - (top + bottom) / (top - bottom);
	M.a20 = sd_zero;
	M.a21 = sd_zero;
	M.a22 = - sd_two / (f - n);
	M.a23 = - (f + n) / (f - n);
	M.a30 = sd_zero;
	M.a31 = sd_zero;
	M.a32 = sd_zero;
	M.a33 = sd_one;
	return M;
}

sd_scalar intersects(const vec3 org, const vec3 raydir, const vec3 sphereOrg, sd_scalar radius, bool discardInside)
{
	// Adjust ray origin relative to sphere center
	const vec3& rayorig = org - sphereOrg;

	// Check origin inside first
	if (rayorig.sq_norm() <= radius*radius && discardInside)
	{
		return 0;
	}

	// Mmm, quadratics
	// Build coeffs which can be used with std quadratic solver
	// ie t = (-b +/- sqrt(b*b + 4ac)) / 2a
	sd_scalar a = dot(raydir, raydir);
	sd_scalar b = 2 * dot(rayorig, raydir);
	sd_scalar c = dot(rayorig, rayorig) - radius*radius;

	// Calc determinant
	sd_scalar d = (b*b) - (4 * a * c);
	if (d < 0)
	{
		// No intersection
		return FLT_MAX;
	}
	else
	{
		// BTW, if d=0 there is one intersection, if d > 0 there are 2
		// But we only want the closest one, so that's ok, just use the 
		// '-' version of the solver
		sd_scalar t = (-b - sqrtf(d)) / (2 * a);
		if (t < 0)
			t = (-b + sqrtf(d)) / (2 * a);
		return t;
	}
}

void getPickRay(int mouseX, int mouseY, mat4 pmatView, mat4 pmatProj, sd_scalar viewPortX, sd_scalar viewPortY, sd_scalar viewPortW, sd_scalar viewPortH, vec3& vPickRayDir, vec3& vPickRayOrig)
{
	// Get the inverse view matrix
	mat4 matView;
	mat4 matWorld;

	// Compute the vector of the Pick ray in screen space
	vec3 v;
	v.x = (((2.0f * (mouseX - viewPortX)) / viewPortW) - 1) / pmatProj.a00;
	v.y = -(((2.0f * (mouseY - viewPortY)) / viewPortH) - 1) / pmatProj.a11;
	v.z = -1.0f;

	mat4 m;
	invert(m, pmatView);
	mat3 m3 = m.get_mat3();
	vPickRayDir = m3 * v;
	// Transform the screen space Pick ray into 3D space
	//vPickRayDir.x = v.x * m.a00 + v.y * m.a01 + v.z * m.a0;
	//vPickRayDir.y = v.x * m._12 + v.y * m._22 + v.z * m._32;
	//vPickRayDir.z = v.x * m._13 + v.y * m._23 + v.z * m._33;
	vPickRayOrig.x = m._14;
	vPickRayOrig.y = m._24;
	vPickRayOrig.z = m._34;
}


const quat quat::Identity(0, 0, 0, 1);

quat::quat()
{
}

quat::quat(sd_scalar x, sd_scalar y, sd_scalar z, sd_scalar w) : x(x), y(y), z(z), w(w)
{
}

quat::quat(const quat& quat)
{
	x = quat.x;
	y = quat.y;
	z = quat.z;
	w = quat.w;
}

quat::quat(const vec3& axis, sd_scalar angle)
{
	sd_scalar len = axis.norm();
	if (len) {
		sd_scalar invLen = 1 / len;
		sd_scalar angle2 = angle / 2;
		sd_scalar scale = sinf(angle2) * invLen;
		x = scale * axis[0];
		y = scale * axis[1];
		z = scale * axis[2];
		w = cosf(angle2);
	}
}

quat::quat(const mat3& rot)
{
	FromMatrix(rot);
}

quat& quat::operator=(const quat& quat)
{
	x = quat.x;
	y = quat.y;
	z = quat.z;
	w = quat.w;
	return *this;
}

quat quat::Inverse()
{
	return quat(- x, - y, - z, w);
}

void quat::Normalize()
{
	sd_scalar len = sqrtf(x * x + y * y + z * z + w * w);
	if (len > 0) {
		sd_scalar invLen = 1 / len;
		x *= invLen;
		y *= invLen;
		z *= invLen;
		w *= invLen;
	}
}

void quat::FromMatrix(const mat3& mat)
{
	sd_scalar trace = mat(0, 0) + mat(1, 1) + mat(2, 2);
	if (trace > sd_zero) 
	{
		sd_scalar scale = sqrtf(trace + sd_one);
		w = sd_zero_5 * scale;
		scale = sd_zero_5 / scale;
		x = scale * (mat(2, 1) - mat(1, 2));
		y = scale * (mat(0, 2) - mat(2, 0));
		z = scale * (mat(1, 0) - mat(0, 1));
	}
	else 
	{
		static int next[] = { 1, 2, 0 };
		int i = 0;
		if (mat(1, 1) > mat(0, 0))
			i = 1;
		if (mat(2, 2) > mat(i, i))
			i = 2;
		int j = next[i];
		int k = next[j];
		sd_scalar scale = sqrtf(mat(i, i) - mat(j, j) - mat(k, k) + 1);
		sd_scalar* q[] = { &x, &y, &z };
		*q[i] = 0.5f * scale;
		scale = 0.5f / scale;
		w = scale * (mat(k, j) - mat(j, k));
		*q[j] = scale * (mat(j, i) + mat(i, j));
		*q[k] = scale * (mat(k, i) + mat(i, k));
	}
}

void quat::ToMatrix(mat3& mat) const
{
	sd_scalar x2 = x * 2;
	sd_scalar y2 = y * 2;
	sd_scalar z2 = z * 2;
	sd_scalar wx = x2 * w;
	sd_scalar wy = y2 * w;
	sd_scalar wz = z2 * w;
	sd_scalar xx = x2 * x;
	sd_scalar xy = y2 * x;
	sd_scalar xz = z2 * x;
	sd_scalar yy = y2 * y;
	sd_scalar yz = z2 * y;
	sd_scalar zz = z2 * z;
	mat(0, 0) = 1 - (yy + zz);
	mat(0, 1) = xy - wz;
	mat(0, 2) = xz + wy;
	mat(1, 0) = xy + wz;
	mat(1, 1) = 1 - (xx + zz);
	mat(1, 2) = yz - wx;
	mat(2, 0) = xz - wy;
	mat(2, 1) = yz + wx;
	mat(2, 2) = 1 - (xx + yy);
}

sd_scalar quat::getEulerAngle()
{
	sd_scalar cangle = w;
	vec3 raxis = GetAxis();
	sd_scalar sangle = raxis.norm();
	raxis.normalize();
	sd_scalar rAngle = atan2(sangle, cangle)*2;

	if ( rAngle > sd_pi )
		rAngle -= sd_two_pi;
	if ( rAngle < -sd_pi )
		rAngle += sd_two_pi;

	return rAngle;
}

vec3 quat::GetAxis()
{
	return vec3(x, y, z);
}

const quat operator*(const quat& p, const quat& q)
{
	return quat(
		p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y,
		p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z,
		p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x,
		p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z
		);
}

quat& quat::operator*=(const quat& q)
{
	*this = *this * q;
	return *this;
}

mat3 & quat_2_mat(mat3& M, const quat& q)
{
	q.ToMatrix(M);
	return M;
}

quat & mat_2_quat(quat& q, const mat3& M)
{
	q.FromMatrix(M);
	return q;
} 

quat & mat_2_quat(quat& q, const mat4& M)
{
	mat3 m;
	M.get_rot(m);
	q.FromMatrix(m);
	return q;
} 

/*
Given an axis and angle, compute quaternion.
*/
quat & axis_to_quat(quat& q, const vec3& a, const sd_scalar phi)
{
	vec3 tmp(a.x, a.y, a.z);

	normalize(tmp);
	sd_scalar s = sinf(phi/sd_two);
	q.x = s * tmp.x;
	q.y = s * tmp.y;
	q.z = s * tmp.z;
	q.w = cosf(phi/sd_two);
	return q;
}

quat & conj(quat & p)
{
	p.x = -p.x;
	p.y = -p.y;
	p.z = -p.z;
	return p;
}

quat & conj(quat& p, const quat& q)
{
	p.x = -q.x;
	p.y = -q.y;
	p.z = -q.z;
	p.w = q.w;
	return p;
}

quat & add_quats(quat& p, const quat& q1, const quat& q2)
{
	quat t1, t2;

	t1 = q1;
	t1.x *= q2.w;
	t1.y *= q2.w;
	t1.z *= q2.w;

	t2 = q2;
	t2.x *= q1.w;
	t2.y *= q1.w;
	t2.z *= q1.w;

	p.x = (q2.y * q1.z) - (q2.z * q1.y) + t1.x + t2.x;
	p.y = (q2.z * q1.x) - (q2.x * q1.z) + t1.y + t2.y;
	p.z = (q2.x * q1.y) - (q2.y * q1.x) + t1.z + t2.z;
	p.w = q1.w * q2.w - (q1.x * q2.x + q1.y * q2.y + q1.z * q2.z);

	return p;
}

sd_scalar & dot(sd_scalar& s, const quat& q1, const quat& q2)
{
	s = q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;
	return s;
}

sd_scalar dot(const quat& q1, const quat& q2)
{
	return q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;
}

#ifndef acosf
#define acosf acos
#endif

quat & slerp_quats(quat & p, sd_scalar s, const quat & q1, const quat & q2)
{
	sd_scalar cosine = dot(q1, q2);
	if (cosine < -1)
		cosine = -1;
	else if (cosine > 1)
		cosine = 1;
	sd_scalar angle = (sd_scalar)acosf(cosine);
	if (fabs(angle) < sd_eps) {
		p = q1;
		return p;
	}
	sd_scalar sine = sinf(angle);
	sd_scalar sineInv = 1.0f / sine;
	sd_scalar c1 = sinf((1.0f - s) * angle) * sineInv;
	sd_scalar c2 = sinf(s * angle) * sineInv;
	p.x = c1 * q1.x + c2 * q2.x;
	p.y = c1 * q1.y + c2 * q2.y;
	p.z = c1 * q1.z + c2 * q2.z;
	p.w = c1 * q1.w + c2 * q2.w;
	return p;
}

const int HALF_RAND = (RAND_MAX / 2);

sd_scalar sd_random()
{
	return ((sd_scalar)(rand() - HALF_RAND) / (sd_scalar)HALF_RAND);
}

// v is normalized
// theta in radians
void mat3::set_rot(const sd_scalar& theta, const vec3& v) 
{
	sd_scalar ct = sd_scalar(cos(theta));
	sd_scalar st = sd_scalar(sin(theta));

	sd_scalar xx = v.x * v.x;
	sd_scalar yy = v.y * v.y;
	sd_scalar zz = v.z * v.z;
	sd_scalar xy = v.x * v.y;
	sd_scalar xz = v.x * v.z;
	sd_scalar yz = v.y * v.z;

	a00 = xx + ct*(1-xx);
	a01 = xy + ct*(-xy) + st*-v.z;
	a02 = xz + ct*(-xz) + st*v.y;

	a10 = xy + ct*(-xy) + st*v.z;
	a11 = yy + ct*(1-yy);
	a12 = yz + ct*(-yz) + st*-v.x;

	a20 = xz + ct*(-xz) + st*-v.y;
	a21 = yz + ct*(-yz) + st*v.x;
	a22 = zz + ct*(1-zz);
}

void mat3::set_rot(const vec3& u, const vec3& v)
{
	sd_scalar phi;
	sd_scalar h;
	sd_scalar lambda;
	vec3 w;

	cross(w,u,v);
	dot(phi,u,v);
	dot(lambda,w,w);
	if (lambda > sd_eps)
		h = (sd_one - phi) / lambda;
	else
		h = lambda;

	sd_scalar hxy = w.x * w.y * h;
	sd_scalar hxz = w.x * w.z * h;
	sd_scalar hyz = w.y * w.z * h;

	a00 = phi + w.x * w.x * h;
	a01 = hxy - w.z;
	a02 = hxz + w.y;

	a10 = hxy + w.z;
	a11 = phi + w.y * w.y * h;
	a12 = hyz - w.x;

	a20 = hxz - w.y;
	a21 = hyz + w.x;
	a22 = phi + w.z * w.z * h;
}

//axis = 0,1,2일때 x,y,z이다.
static mat3 RotationMatrix(sd_scalar sine, sd_scalar cosine, int axis)
{
	mat3 mat;
	if (axis == 0)
	{
		mat.set_row(0, vec3(1.0, 0.0, 0.0));
		mat.set_row(1, vec3(0.0, cosine, -sine));
		mat.set_row(2, vec3(0.0, sine, cosine));
	}
	else if (axis == 1)
	{
		mat.set_row(0, vec3(cosine, 0.0, sine));
		mat.set_row(1, vec3(0.0, 1.0, 0.0));
		mat.set_row(2, vec3(-sine, 0.0, cosine));
	}
	else
	{
		mat.set_row(0, vec3(cosine, -sine, 0.0));
		mat.set_row(1, vec3(sine, cosine, 0.0));
		mat.set_row(2, vec3(0.0, 0.0, 1.0));
	}
	return mat;
}

static mat3 RotationMatrix(sd_scalar angle, int axis)
{
	return RotationMatrix(sin(angle), cos(angle), axis);
}

void mat3::set_rot(const vec3 & u)
{
	sd_scalar alpha = u.x;
	sd_scalar beta = u.y;
	sd_scalar gamma = u.z;

	mat3 rot_x = RotationMatrix(alpha, 0);
	mat3 rot_y = RotationMatrix(beta, 1);
	mat3 rot_z = RotationMatrix(gamma, 2);

	mat3 _result = mult(mult(rot_x, rot_y), rot_z);
	memcpy(mat_array, _result.mat_array, sizeof(sd_scalar)*9);
}

void mat3::get_rot(vec3& u)
{
	// Convert rotation matrix to X-Y-Z Euler Angles (alpha, beta, gamma)
	// beta = atan2(r13, sqrt(r11^2+r12^2))
	// alpha = atan2(-r23/cosb, r33/cosb)
	// gamma = atan2(-r12/cosb, r11/cosb)
    u[1] = atan2(a02, sqrt(a00*a00+a01*a01));
    sd_scalar cosb = cos(u[1]);
    u[0] = atan2(-a12 / cosb, a22 / cosb);
    u[2] = atan2(-a01 / cosb, a00 / cosb);
}

void mat3::set_rot_zxy(const vec3 & u)
{
	sd_scalar alpha = u.x;
	sd_scalar beta = u.y;
	sd_scalar gamma = u.z;

	mat3 rot_z = RotationMatrix(alpha, 2);
	mat3 rot_x = RotationMatrix(beta, 0);
	mat3 rot_y = RotationMatrix(gamma, 1);

	mat3 _result = mult(mult(rot_z, rot_x), rot_y);
	memcpy(mat_array, _result.mat_array, sizeof(sd_scalar)*9);
}

void mat3::get_rot_zxy(vec3& u)
{
	// Convert rotation matrix to Z-X-Y Euler Angles.
	// pitch = atan2(r32, sqrt(r12^2+r22^2))
	// roll = atan2(-r12/cosr, r22/cosr)
	// yaw = atan2(-r31/cosr, r33/cosr)
	u[1] = atan2(a21, sqrt(a01*a01+a11*a11));
	double cosr = cos(u[1]);
	u[0] = atan2(-a01 / cosr, a11 / cosr);
	u[2] = atan2(-a20 / cosr, a22 / cosr);
}

void mat3::set_rot_zyx(const vec3 & u)
{
	sd_scalar alpha = u.x;
	sd_scalar beta = u.y;
	sd_scalar gamma = u.z;

	mat3 rot_z = RotationMatrix(alpha, 2);
	mat3 rot_y = RotationMatrix(beta, 1);
	mat3 rot_x = RotationMatrix(gamma, 0);

	mat3 _result = mult(mult(rot_z, rot_y), rot_x);
	memcpy(mat_array, _result.mat_array, sizeof(sd_scalar)*9);
}

void mat3::get_rot_zyx(vec3& u)
{
	// Convert rotation matrix to Z-Y-X Euler Angles (alpha, beta, gamma)
	// beta = atan2(-r31, sqrt(r11^2+r21^2))
	// alpha = atan2(r21/cosb, r11/cosb)
	// gamma = atan2(r32/cosb, r33/cosb)
	u[1] = atan2(-a20, sqrt(a00*a00+a10*a10));
	double cosb = cos(u[1]);
	u[0] = atan2(a10 / cosb, a00 / cosb);
	u[2] = atan2(a21 / cosb, a22 / cosb);
}

void mat3::set_rot_yxz(const vec3& u)
{
	sd_scalar alpha = u.x;
	sd_scalar beta = u.y;
	sd_scalar gamma = u.z;

	mat3 rot_y = RotationMatrix(alpha, 1);
	mat3 rot_x = RotationMatrix(beta, 0);
	mat3 rot_z = RotationMatrix(gamma, 2);

	mat3 _result = mult(mult(rot_y, rot_x), rot_z);
	memcpy(mat_array, _result.mat_array, sizeof(sd_scalar)* 9);
}

void mat3::get_rot_yxz(vec3& u)
{
	// rot =  cy*cz+sx*sy*sz  cz*sx*sy-cy*sz  cx*sy
	//        cx*sz           cx*cz          -sx
	//       -cz*sy+cy*sx*sz  cy*cz*sx+sy*sz  cx*cy

	u[1] = asin(-a12);
	if( u[1] < sd_half_pi)
	{
		if (u[1] > -sd_half_pi)
		{
			u[0] = atan2(a02, a22);
			u[2] = atan2(a10, a11);
		}
		else
		{
			// WARNING.  Not a unique solution.
			u[2] = 0.0;  // any angle works
			u[0] = -atan2(-a01, a00);
		}
	}
	else
	{
		// WARNING.  Not a unique solution.
		u[2] = 0.0; // any angle works
		u[0] = atan2(-a01, a00);;
	}
}
void mat3::set_rodrigues(const vec3& u)
{
	int k;
	sd_scalar rx = u[0], ry = u[1], rz = u[2];
	sd_scalar theta = u.norm();

	if( theta < FLT_EPSILON )
	{
		*this = mat3_id;
	}
	else
	{
		const sd_scalar I[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

		sd_scalar c = cos(theta);
		sd_scalar s = sin(theta);
		sd_scalar c1 = 1. - c;
		sd_scalar itheta = theta ? 1./theta : 0.;

		rx *= itheta; ry *= itheta; rz *= itheta;

		sd_scalar rrt[] = { rx*rx, rx*ry, rx*rz, rx*ry, ry*ry, ry*rz, rx*rz, ry*rz, rz*rz };
		sd_scalar _r_x_[] = { 0, -rz, ry, rz, 0, -rx, -ry, rx, 0 };

		// R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
		// where [r_x] is [0 -rz ry; rz 0 -rx; -ry rx 0]
		for( k = 0; k < 9; k++ )			
			mat_array[k] = c*I[k] + c1*rrt[k] + s*_r_x_[k];
		transpose(*this);
	}
}

vec3 mat3::get_rodrigues()
{
	//sin(theta)*vecmul(r) = (R-R')/2
	// theta = cos-1( (r11+r22+r33-1)/2 )
	// r = 1/(2*sin(theta)) * [r32-r23; r13-r31; r21-r12];
	vec3 r;
	r[0] = -(a12-a21) / 2;
	r[1] = (a02-a20) / 2;
	r[2] = -(a01-a10) / 2;
	sd_scalar sint = r.norm();
		
	vec3 res;
	sd_scalar theta;
	if(sint<DBL_EPSILON)
	{
		res = vec3_null;
	}
	else
	{
		for( int i = 0; i < 3; i++ )
			r[i] /= sint;
		if(abs(r[0]-1)<DBL_EPSILON)
			theta = 1;
		else
		{
			double cost = (a00 - r[0]*r[0]) / (1 - r[0]*r[0]);
			if(cost>0)
				theta = asin(sint);
			else
				theta = sd_pi - asin(sint);
		}
			
		for( int i = 0; i < 3; i++ )
			res[i] = r[i] * theta;
	}
	return res;
}

void mat3::ortho()
{
	vec3 v1 = col(0);
	vec3 v2 = col(1);
	vec3 v3 = col(2);
	v1.normalize();
	v2.normalize();
	v3.normalize();
	set_col(0, v1);
	set_col(1, v2);
	set_col(2, v3);
}

sd_scalar mat3::norm_one()
{
	sd_scalar sum, max = sd_zero;
	sum = fabs(a00) + fabs(a10) + fabs(a20);
	if (max < sum)
		max = sum;
	sum = fabs(a01) + fabs(a11) + fabs(a21);
	if (max < sum)
		max = sum;
	sum = fabs(a02) + fabs(a12) + fabs(a22);
	if (max < sum)
		max = sum;
	return max;
}

sd_scalar mat3::norm_inf()
{
	sd_scalar sum, max = sd_zero;
	sum = fabs(a00) + fabs(a01) + fabs(a02);
	if (max < sum)
		max = sum;
	sum = fabs(a10) + fabs(a11) + fabs(a12);
	if (max < sum)
		max = sum;
	sum = fabs(a20) + fabs(a21) + fabs(a22);
	if (max < sum)
		max = sum;
	return max;
}

void mat4::set_mat3(mat3 & mat)
{
	a00 = mat.a00;
	a10 = mat.a10;
	a20 = mat.a20;

	a01 = mat.a01;
	a11 = mat.a11;
	a21 = mat.a21;

	a02 = mat.a02;
	a12 = mat.a12;
	a22 = mat.a22;

	/*	Reevan's Bugfix */
	a30=0.0F;
	a31=0.0F;
	a32=0.0F;

	a03=0.0F;	
	a13=0.0F;	
	a23=0.0F;	
	a33=1.0F;
}

const mat3 mat4::get_mat3()
{
	mat3 mat;
	mat.a00 = a00;
	mat.a10 = a10;
	mat.a20 = a20;

	mat.a01 = a01;
	mat.a11 = a11;
	mat.a21 = a21;

	mat.a02 = a02;
	mat.a12 = a12;
	mat.a22 = a22;
	return mat;
}

void mat4::set_rot(const quat& q)
{
	mat3 m;
	q.ToMatrix(m);
	set_rot(m);
}

bool sd_isZero(sd_scalar x)
{
	sd_scalar limit = sd_eps;
	if (x < limit && x > -limit)
		return true;
	else
		return false;
}

// v is normalized
// theta in radians
void mat4::set_rot(const sd_scalar& theta, const vec3& v) 
{
	sd_scalar ct = sd_scalar(cos(theta));
	sd_scalar st = sd_scalar(sin(theta));

	sd_scalar xx = v.x * v.x;
	sd_scalar yy = v.y * v.y;
	sd_scalar zz = v.z * v.z;
	sd_scalar xy = v.x * v.y;
	sd_scalar xz = v.x * v.z;
	sd_scalar yz = v.y * v.z;

	a00 = xx + ct*(1-xx);
	a01 = xy + ct*(-xy) + st*-v.z;
	a02 = xz + ct*(-xz) + st*v.y;

	a10 = xy + ct*(-xy) + st*v.z;
	a11 = yy + ct*(1-yy);
	a12 = yz + ct*(-yz) + st*-v.x;

	a20 = xz + ct*(-xz) + st*-v.y;
	a21 = yz + ct*(-yz) + st*v.x;
	a22 = zz + ct*(1-zz);
}
void mat4::zeroProcess()
{
}

void mat4::set_rot(const vec3& u, const vec3& v)
{
	sd_scalar phi;
	sd_scalar h;
	sd_scalar lambda;
	vec3 w;

	cross(w,u,v);
	dot(phi,u,v);
	dot(lambda,w,w);
	if (lambda > sd_eps)
		h = (sd_one - phi) / lambda;
	else
		h = lambda;

	sd_scalar hxy = w.x * w.y * h;
	sd_scalar hxz = w.x * w.z * h;
	sd_scalar hyz = w.y * w.z * h;

	a00 = phi + w.x * w.x * h;
	a01 = hxy - w.z;
	a02 = hxz + w.y;

	a10 = hxy + w.z;
	a11 = phi + w.y * w.y * h;
	a12 = hyz - w.x;

	a20 = hxz - w.y;
	a21 = hyz + w.x;
	a22 = phi + w.z * w.z * h;
}

void mat4::set_rot(const vec3& u)
{
    double alpha = u.x;
    double beta = u.y;
    double gamma = u.z;
    double ca = cos(alpha);
    double sa = sin(alpha);
    double cb = cos(beta);
    double sb = sin(beta);
    double cr = cos(gamma);
    double sr = sin(gamma);

    a00 = cb*cr;
    a01 = -cb*sr;
    a02 = sb;
    a10 = ca*sr+cr*sa*sb;
    a11 = ca*cr-sa*sb*sr;
    a12 = -cb*sa;
    a20 = sa*sr-ca*sb*cr;
    a21 = cr*sa+ca*sb*sr;
    a22 = ca*cb;
}

void mat4::get_rot(vec3& u)
{
    u[1] = atan2(a02, sqrt(a00*a00+a01*a01));
    double cosb = cos(u[1]);
// 	if (sd_isZero(cosb))
// 		cosb = 1.0f;
    u[0] = atan2(-a12 / cosb, a22 / cosb);
    u[2] = atan2(-a01 / cosb, a00 / cosb);
}

void mat4::set_rot(const mat3& M)
{
	// copy the 3x3 rotation block
	a00 = M.a00; a10 = M.a10; a20 = M.a20;
	a01 = M.a01; a11 = M.a11; a21 = M.a21;
	a02 = M.a02; a12 = M.a12; a22 = M.a22;
}

void mat4::set_scale(const vec3& s)
{
	a00 = s.x;
	a11 = s.y;
	a22 = s.z;
}

vec3& mat4::get_scale(vec3& s) const
{
	s.x = a00;
	s.y = a11;
	s.z = a22;
	return s;
}

void mat4::set_translation(const vec3& t)
{
	a03 = t.x;
	a13 = t.y;
	a23 = t.z;
}

vec3 & mat4::get_translation(vec3& t) const
{
	t.x = a03;
	t.y = a13;
	t.z = a23;
	return t;
}

mat3 & mat4::get_rot(mat3& M) const
{
	// assign the 3x3 rotation block
	M.a00 = a00; M.a10 = a10; M.a20 = a20;
	M.a01 = a01; M.a11 = a11; M.a21 = a21;
	M.a02 = a02; M.a12 = a12; M.a22 = a22;
	return M;
}

quat & mat4::get_rot(quat& q) const
{
	mat3 m;
	get_rot(m);
	q.FromMatrix(m);
	return q;
}

mat4 & negate(mat4 & M)
{
	for (int i = 0; i < 16; ++i)
		M.mat_array[i]= -M.mat_array[i];
	return M;
}

mat3 & negate(mat3 & M)
{
	for (int i = 0; i < 9; ++i)
		M.mat_array[i]= -M.mat_array[i];
	return M;
}

mat3& tangent_basis(mat3& basis, const vec3& v0, const vec3& v1, const vec3& v2, const vec2& t0, const vec2& t1, const vec2& t2, const vec3 & n)
{
	vec3 cp;
	vec3 e0(v1.x - v0.x, t1.s - t0.s, t1.t - t0.t);
	vec3 e1(v2.x - v0.x, t2.s - t0.s, t2.t - t0.t);

	cross(cp,e0,e1);
	if ( fabs(cp.x) > sd_eps)
	{
		basis.a00 = -cp.y / cp.x;        
		basis.a10 = -cp.z / cp.x;
	}

	e0.x = v1.y - v0.y;
	e1.x = v2.y - v0.y;

	cross(cp,e0,e1);
	if ( fabs(cp.x) > sd_eps)
	{
		basis.a01 = -cp.y / cp.x;        
		basis.a11 = -cp.z / cp.x;
	}

	e0.x = v1.z - v0.z;
	e1.x = v2.z - v0.z;

	cross(cp,e0,e1);
	if ( fabs(cp.x) > sd_eps)
	{
		basis.a02 = -cp.y / cp.x;        
		basis.a12 = -cp.z / cp.x;
	}

	// tangent...
	sd_scalar oonorm = sd_one / sqrtf(basis.a00 * basis.a00 + basis.a01 * basis.a01 + basis.a02 * basis.a02);
	basis.a00 *= oonorm;
	basis.a01 *= oonorm;
	basis.a02 *= oonorm;

	// binormal...
	oonorm = sd_one / sqrtf(basis.a10 * basis.a10 + basis.a11 * basis.a11 + basis.a12 * basis.a12);
	basis.a10 *= oonorm;
	basis.a11 *= oonorm;
	basis.a12 *= oonorm;

	// normal...
	// compute the cross product TxB
	basis.a20 = basis.a01*basis.a12 - basis.a02*basis.a11;
	basis.a21 = basis.a02*basis.a10 - basis.a00*basis.a12;
	basis.a22 = basis.a00*basis.a11 - basis.a01*basis.a10;

	oonorm = sd_one / sqrtf(basis.a20 * basis.a20 + basis.a21 * basis.a21 + basis.a22 * basis.a22);
	basis.a20 *= oonorm;
	basis.a21 *= oonorm;
	basis.a22 *= oonorm;

	// Gram-Schmidt orthogonalization process for B
	// compute the cross product B=NxT to obtain 
	// an orthogonal basis
	basis.a10 = basis.a21*basis.a02 - basis.a22*basis.a01;
	basis.a11 = basis.a22*basis.a00 - basis.a20*basis.a02;
	basis.a12 = basis.a20*basis.a01 - basis.a21*basis.a00;

	if (basis.a20 * n.x + basis.a21 * n.y + basis.a22 * n.z < sd_zero)
	{
		basis.a20 = -basis.a20;
		basis.a21 = -basis.a21;
		basis.a22 = -basis.a22;
	}
	return basis;
}

/*
* Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
* if we are away from the center of the sphere.
*/
sd_scalar tb_project_to_sphere(sd_scalar r, sd_scalar x, sd_scalar y)
{
	sd_scalar d, t, z;

	d = sqrtf(x*x + y*y);
	if (d < r * 0.70710678118654752440) {    /* Inside sphere */
		z = sqrtf(r*r - d*d);
	} else {           /* On hyperbola */
		t = r / (sd_scalar)1.41421356237309504880;
		z = t*t / d;
	}
	return z;
}

/*
* Ok, simulate a track-ball.  Project the points onto the virtual
* trackball, then figure out the axis of rotation, which is the cross
* product of P1 P2 and O P1 (O is the center of the ball, 0,0,0)
* Note:  This is a deformed trackball-- is a trackball in the center,
* but is deformed into a hyperbolic sheet of rotation away from the
* center.  This particular function was chosen after trying out
* several variations.
*
* It is assumed that the arguments to this routine are in the range
* (-1.0 ... 1.0)
*/
quat & trackball(quat& q, vec2& pt1, vec2& pt2, sd_scalar trackballsize)
{
	vec3 a; // Axis of rotation
	sd_scalar phi;  // how much to rotate about axis
	vec3 d;
	sd_scalar t;

	if (pt1.x == pt2.x && pt1.y == pt2.y) 
	{
		// Zero rotation
		q = quat_id;
		return q;
	}

	// First, figure out z-coordinates for projection of P1 and P2 to
	// deformed sphere
	vec3 p1(pt1.x,pt1.y,tb_project_to_sphere(trackballsize,pt1.x,pt1.y));
	vec3 p2(pt2.x,pt2.y,tb_project_to_sphere(trackballsize,pt2.x,pt2.y));

	//  Now, we want the cross product of P1 and P2
	cross(a,p1,p2);

	//  Figure out how much to rotate around that axis.
	d.x = p1.x - p2.x;
	d.y = p1.y - p2.y;
	d.z = p1.z - p2.z;
	t = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z) / (trackballsize);

	// Avoid problems with out-of-control values...

	if (t > sd_one)
		t = sd_one;
	if (t < -sd_one) 
		t = -sd_one;
	phi = sd_two * sd_scalar(asin(t));
	axis_to_quat(q,a,phi);
	return q;
}

vec3& cube_map_normal(int i, int x, int y, int cubesize, vec3& v)
{
	sd_scalar s, t, sc, tc;
	s = (sd_scalar(x) + sd_zero_5) / sd_scalar(cubesize);
	t = (sd_scalar(y) + sd_zero_5) / sd_scalar(cubesize);
	sc = s * sd_two - sd_one;
	tc = t * sd_two - sd_one;

	switch (i) 
	{
	case 0:
		v.x = sd_one;
		v.y = -tc;
		v.z = -sc;
		break;
	case 1:
		v.x = -sd_one;
		v.y = -tc;
		v.z = sc;
		break;
	case 2:
		v.x = sc;
		v.y = sd_one;
		v.z = tc;
		break;
	case 3:
		v.x = sc;
		v.y = -sd_one;
		v.z = -tc;
		break;
	case 4:
		v.x = sc;
		v.y = -tc;
		v.z = sd_one;
		break;
	case 5:
		v.x = -sc;
		v.y = -tc;
		v.z = -sd_one;
		break;
	}
	normalize(v);
	return v;
}

// computes the area of a triangle
sd_scalar sd_area(const vec3& v1, const vec3& v2, const vec3& v3)
{
	vec3 cp_sum;
	vec3 cp;
	cross(cp_sum, v1, v2);
	cp_sum += cross(cp, v2, v3);
	cp_sum += cross(cp, v3, v1);
	return sd_norm(cp_sum) * sd_zero_5; 
}

// computes the perimeter of a triangle
sd_scalar sd_perimeter(const vec3& v1, const vec3& v2, const vec3& v3)
{
	sd_scalar perim;
	vec3 diff;
	sub(diff, v1, v2);
	perim = sd_norm(diff);
	sub(diff, v2, v3);
	perim += sd_norm(diff);
	sub(diff, v3, v1);
	perim += sd_norm(diff);
	return perim;
}

// compute the center and radius of the inscribed circle defined by the three vertices
sd_scalar sd_find_in_circle(vec3& center, const vec3& v1, const vec3& v2, const vec3& v3)
{
	sd_scalar area = sd_area(v1, v2, v3);
	// if the area is null
	if (area < sd_eps)
	{
		center = v1;
		return sd_zero;
	}

	sd_scalar oo_perim = sd_one / sd_perimeter(v1, v2, v3);

	vec3 diff;

	sub(diff, v2, v3);
	mult(center, v1, sd_norm(diff));

	sub(diff, v3, v1);
	madd(center, v2, sd_norm(diff));

	sub(diff, v1, v2);
	madd(center, v3, sd_norm(diff));

	center *= oo_perim;

	return sd_two * area * oo_perim;
}

// compute the center and radius of the circumscribed circle defined by the three vertices
// i.e. the osculating circle of the three vertices
sd_scalar sd_find_circ_circle( vec3& center, const vec3& v1, const vec3& v2, const vec3& v3)
{
	vec3 e0;
	vec3 e1;
	sd_scalar d1, d2, d3;
	sd_scalar c1, c2, c3, oo_c;

	sub(e0, v3, v1);
	sub(e1, v2, v1);
	dot(d1, e0, e1);

	sub(e0, v3, v2);
	sub(e1, v1, v2);
	dot(d2, e0, e1);

	sub(e0, v1, v3);
	sub(e1, v2, v3);
	dot(d3, e0, e1);

	c1 = d2 * d3;
	c2 = d3 * d1;
	c3 = d1 * d2;
	oo_c = sd_one / (c1 + c2 + c3);

	mult(center,v1,c2 + c3);
	madd(center,v2,c3 + c1);
	madd(center,v3,c1 + c2);
	center *= oo_c * sd_zero_5;

	return sd_zero_5 * sqrtf((d1 + d2) * (d2 + d3) * (d3 + d1) * oo_c);
}

sd_scalar ffast_cos(const sd_scalar x)
{
	// assert:  0 <= fT <= PI/2
	// maximum absolute error = 1.1880e-03
	// speedup = 2.14

	sd_scalar x_sqr = x*x;
	sd_scalar res = sd_scalar(3.705e-02);
	res *= x_sqr;
	res -= sd_scalar(4.967e-01);
	res *= x_sqr;
	res += sd_one;
	return res;
}


sd_scalar fast_cos(const sd_scalar x)
{
	// assert:  0 <= fT <= PI/2
	// maximum absolute error = 2.3082e-09
	// speedup = 1.47

	sd_scalar x_sqr = x*x;
	sd_scalar res = sd_scalar(-2.605e-07);
	res *= x_sqr;
	res += sd_scalar(2.47609e-05);
	res *= x_sqr;
	res -= sd_scalar(1.3888397e-03);
	res *= x_sqr;
	res += sd_scalar(4.16666418e-02);
	res *= x_sqr;
	res -= sd_scalar(4.999999963e-01);
	res *= x_sqr;
	res += sd_one;
	return res;
}

void sd_is_valid(const vec3& v)
{
	assert(!_isnan(v.x) && !_isnan(v.y) && !_isnan(v.z) &&
		_finite(v.x) && _finite(v.y) && _finite(v.z));
}

void sd_is_valid(sd_scalar lambda)
{
	assert(!_isnan(lambda) && _finite(lambda));
}

void get_rot(const vec3& i, const vec3& u, const vec3& v, sd_scalar& theta, vec3& axis)
{
	cross(axis, u, v);
	axis.normalize();
	
	if ( axis.norm() < sd_eps )
	{
		cross(axis, i, v);
		axis.normalize();
		if(axis.norm() < sd_eps)
		{
			cross(axis, vec3_z, v);
			axis.normalize();
			/*if(axis.norm() < sd_eps)
			{
				cross(axis, vec3_z, u);
				axis.normalize();
				theta = 0;
				return;
			}*/
		}
	}

	vec3 nu = u; nu.normalize();
	vec3 nv = v; nv.normalize();
	sd_scalar d = dot(nu, nv);
	if ( d > 1 )
		d = 1;
	if ( d < -1 )
		d = -1;
	theta = acos(d);

	if ( theta > sd_pi )
	{
		theta = sd_two_pi-theta;
		axis *= -1;
	}
	else if ( theta < -sd_pi )
	{
		theta = sd_two_pi+theta;
		axis *= -1;
	}
}

float interp(float start, float end, float current, int type, bool checkBound )
{
	if ( checkBound && current >= end )
		return 1;
	if ( checkBound && current <= start )
		return 0;

	float s = 0;
	float t = (current-start)/(end - start);
	switch ( type )
	{
	case 1://strike
		{
			float a = 0.6f, th = 0.9f,q=0.4f;
			if (t <= th*q)
				s = a/th*t/q;
			else if (t < q) {
				float b = (t/q-th)/(1-th);
				s = a + (1-a) * b * b * b * b * b;
			}
			else
				s = 1.0f;
		}
		break;
	case 2://push
		{
/*
			float a = 0.4f, th = 0.1f;
			if (t > th)
				s = a + (1-a)*(t-th)/(1-th);
			else {
				//float b = t/th;
				//s = a * b * b * b * b;
				s = a * t/th;
			}
*/
 			QEasingCurve curve(QEasingCurve::OutQuint);
 			s = curve.valueForProgress( t );
		}
		break;
	case 3: //fast  ( s = v * t + 0.5*a*t*t )[v = 1, a = 1]
			s = 0.5 * t + 0.5 * t * t * t * t;
		break;
	case 4: //slow
			s = 1.5 * t - 0.5 * t * t * t;
		break;
	case 0://linear
	default:
		QEasingCurve curve(QEasingCurve::Linear);
		s = curve.valueForProgress( t );
	}

	return s;
}

mat3 getRot(vec3 preLeft, vec3 preUp, vec3 afterLeft, vec3 afterUp)
{
	vec3 preForward, afterForward;
	cross(preForward, preLeft, preUp);
	cross(preUp, preForward, preLeft);
	cross(afterForward, afterLeft, afterUp);
	cross(afterUp, afterForward, afterLeft);

	preLeft.normalize();
	preUp.normalize();
	preForward.normalize();
	afterLeft.normalize();
	afterUp.normalize();
	afterForward.normalize();

	mat3 mat_rot, rot_target, rot_subject;
	rot_target.set_col(0, afterLeft);
	rot_target.set_col(1, afterUp);
	rot_target.set_col(2, afterForward);
	rot_subject.set_col(0, preLeft);
	rot_subject.set_col(1, preUp);
	rot_subject.set_col(2, preForward);
	mat3 rot_subject_transp;
	transpose(rot_subject_transp, rot_subject);
	mat_rot = mult(rot_target, rot_subject_transp); // world rotation
	return mat_rot;
}

bool IntersectTriangle( const vec3 orig, const vec3 dir, //ray
	vec3 v0, vec3 v1, vec3 v2, //triangle
	float* t, //Ray-intersection parameter distance.
	float* u, //Barycentric hit coordinates, U.
	float* v )//Barycentric hit coordinates, V.
	//--------------------------------------------------------------------------------------
	// Given a ray origin (orig) and direction (dir), and three vertices of a triangle, this
	// function returns true and the interpolated texture coordinates if the ray intersects 
	// the triangle
	//--------------------------------------------------------------------------------------
{
	// Find vectors for two edges sharing vert0
	vec3 edge1;
	edge1[0] = v1[0] - v0[0];
	edge1[1] = v1[1] - v0[1];
	edge1[2] = v1[2] - v0[2];

	vec3 edge2;
	edge2[0] = v2[0] - v0[0];
	edge2[1] = v2[1] - v0[1];
	edge2[2] = v2[2] - v0[2];

	// Begin calculating determinant - also used to calculate U parameter
	vec3 pvec;
	cross(pvec, dir, edge2);

	// If determinant is near zero, ray lies in plane of triangle
	sd_scalar det = dot(edge1, pvec);

	vec3 tvec;
	if( det > 0 )
	{
		sub(tvec, orig, v0);
	}
	else
	{
		sub(tvec, v0, orig);
		det = -det;
	}

	if( det < 0.0001f )
		return false;

	// Calculate U parameter and test bounds
	*u = dot( tvec, pvec );
	if( *u < 0.0f || *u > det )
		return false;

	// Prepare to test V parameter
	vec3 qvec;
	cross(qvec, tvec , edge1);

	// Calculate V parameter and test bounds
	*v = dot( dir, qvec );
	if( *v < 0.0f || *u + *v > det )
		return false;

	// Calculate t, scale parameters, ray intersects triangle
	*t = dot( edge2, qvec );
	sd_scalar fInvDet = 1.0f / det;
	*t *= fInvDet;
	*u *= fInvDet;
	*v *= fInvDet;

	if ( *t < 0 )
		return false;

	return true;
}

void dropA2BWithLen(vec3 &A, vec3 B, sd_scalar len)
{
	A.y = B.y + sqrt( len*len - (A.x-B.x)*(A.x-B.x) - (A.z-B.z)*(A.z-B.z) );
}

void clip_limit(sd_scalar &val, sd_scalar min, sd_scalar max)
{
	if ( val < min )
		val = min;

	if ( val > max )
		val = max;
}
