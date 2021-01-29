#ifndef _3DMATH_H_
#define _3DMATH_H_

#include <assert.h>
#include <math.h>

#ifdef _WIN32
#include <limits>
#else
#include <limits.h>
#endif

#include <memory.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>

typedef float sd_scalar; //see also the composdata.h

#define sd_zero			      sd_scalar(0)
#define sd_zero_5             sd_scalar(0.5)
#define sd_one			      sd_scalar(1.0)
#define sd_two			      sd_scalar(2)
#define sd_half_pi            sd_scalar(3.14159265358979323846264338327950288419716939937510582 * 0.5)
#define sd_quarter_pi         sd_scalar(3.14159265358979323846264338327950288419716939937510582 * 0.25)
#define sd_pi			      sd_scalar(3.14159265358979323846264338327950288419716939937510582)
#define sd_two_pi			  sd_scalar(3.14159265358979323846264338327950288419716939937510582 * 2.0)
#define sd_oo_pi			  sd_one / sd_pi
#define sd_oo_two_pi	      sd_one / sd_two_pi
#define sd_oo_255   	      sd_one / sd_scalar(255)
#define sd_oo_128   	      sd_one / sd_scalar(128)
#define sd_to_rad             sd_pi / sd_scalar(180)
#define sd_to_deg             sd_scalar(180) / sd_pi
#define sd_eps		          sd_scalar(10e-6)
#define sd_double_eps	      sd_scalar(10e-6) * sd_two
#define sd_big_eps            sd_scalar(10e-2)
#define sd_small_eps          sd_scalar(10e-6)
#define sd_sqrthalf           sd_scalar(0.7071067811865475244)

#define sd_scalar_max         sd_scalar(FLT_MAX)
#define sd_scalar_min         sd_scalar(FLT_MIN)

bool	sd_isZero(sd_scalar x);

struct vec2;
struct vec2t;
struct vec3;
struct vec3t;
struct vec4;
struct vec4t;

struct vec2
{
	vec2() : x(0), y(0) { }
	vec2(sd_scalar x, sd_scalar y) : x(x), y(y) { }
	vec2(const sd_scalar* xy) : x(xy[0]), y(xy[1]) { }
	vec2(const vec2& u) : x(u.x), y(u.y) { }
	vec2(const vec3&);

	bool operator==(const vec2 & u) const
	{
		return (u.x == x && u.y == y) ? true : false;
	}

	bool operator!=(const vec2 & u) const
	{
		return !(*this == u );
	}

	vec2 & operator*=(const sd_scalar & lambda)
	{
		x*= lambda;
		y*= lambda;
		return *this;
	}

	vec2 & operator-=(const vec2 & u)
	{
		x-= u.x;
		y-= u.y;
		return *this;
	}

	vec2 & operator+=(const vec2 & u)
	{
		x+= u.x;
		y+= u.y;
		return *this;
	}

	sd_scalar & operator[](int i)
	{
		return vec_array[i];
	}

	const sd_scalar operator[](int i) const
	{
		return vec_array[i];
	}

	vec2 operator - () const
	{
		return vec2(-x, -y);
	}

	sd_scalar sq_norm() const { return x * x + y * y; }
	sd_scalar norm() const { return sqrtf(sq_norm()); }
	sd_scalar normalize();

	union {
		struct {
			sd_scalar x,y;          // standard names for components
		};
		struct {
			sd_scalar s,t;          // standard names for components
		};
		sd_scalar vec_array[2];     // array access
	};
};

inline const vec2 operator+(const vec2& u, const vec2& v)
{
	return vec2(u.x + v.x, u.y + v.y);
}

inline const vec2 operator-(const vec2& u, const vec2& v)
{
	return vec2(u.x - v.x, u.y - v.y);
}

inline const vec2 operator*(const sd_scalar s, const vec2& u)
{
	return vec2(s * u.x, s * u.y);
}

inline const vec2 operator/(const vec2& u, const sd_scalar s)
{
	return vec2(u.x / s, u.y / s);
}

inline const vec2 operator*(const vec2&u, const vec2&v)
{
	return vec2(u.x * v.x, u.y * v.y);
}

struct vec3
{
	vec3() : x(0), y(0), z(0) { }
	vec3(sd_scalar x, sd_scalar y, sd_scalar z) : x(x), y(y), z(z) { }
	vec3(const sd_scalar* xyz) : x(xyz[0]), y(xyz[1]), z(xyz[2]) { }
	vec3(const vec2& u) : x(u.x), y(u.y), z(1.0f) { }
	vec3(const vec3& u) : x(u.x), y(u.y), z(u.z) { }
	vec3(const vec4&);

	bool operator==(const vec3 & u) const
	{
		return (u.x == x && u.y == y && u.z == z) ? true : false;
	}

	bool operator!=( const vec3& rhs ) const
	{
		return !(*this == rhs );
	}

	vec3 & operator*=(const sd_scalar & lambda)
	{
		x*= lambda;
		y*= lambda;
		z*= lambda;
		return *this;
	}

	vec3 operator - () const
	{
		return vec3(-x, -y, -z);
	}

	vec3 & operator-=(const vec3 & u)
	{
		x-= u.x;
		y-= u.y;
		z-= u.z;
		return *this;
	}

	vec3 & operator+=(const vec3 & u)
	{
		x+= u.x;
		y+= u.y;
		z+= u.z;
		return *this;
	}

	vec3 operator * (const sd_scalar & f) const
	{
		return vec3(x*f, y*f, z*f);
	}

	sd_scalar normalize();
	void orthogonalize( const vec3& v );
	void orthonormalize( const vec3& v )
	{
		orthogonalize( v ); //  just orthogonalize...
		normalize();        //  ...and normalize it
	}

	sd_scalar sq_norm() const { return x * x + y * y + z * z; }
	sd_scalar norm() const { return sqrt(sq_norm()); }

	sd_scalar & operator[](int i)
	{
		return vec_array[i];
	}

	const sd_scalar operator[](int i) const
	{
		return vec_array[i];
	}

	const bool isNaN() const
	{
		bool nan = false;
		for( int i = 0; i < 3; i++ )
			if(_isnan(vec_array[i]) != 0)
			{
				nan = true;
				break;
			}
		if(nan)
		{
		}
		return nan;
	}

	union {
		struct {
			sd_scalar x,y,z;        // standard names for components
		};
		struct {
			sd_scalar s,t,r;        // standard names for components
		};
		sd_scalar vec_array[3];     // array access
	};
};

inline const vec3 operator+(const vec3& u, const vec3& v)
{
	return vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

inline const vec3 operator-(const vec3& u, const vec3& v)
{
	return vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

inline const vec3 operator^(const vec3& u, const vec3& v)
{
	return vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
}

inline const vec3 operator*(const sd_scalar s, const vec3& u)
{
	return vec3(s * u.x, s * u.y, s * u.z);
}

inline const vec3 operator/(const vec3& u, const sd_scalar s)
{
	return vec3(u.x / s, u.y / s, u.z / s);
}

inline const vec3 operator*(const vec3& u, const vec3& v)
{
	return vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

inline vec2::vec2(const vec3& u)
{
	x = u.x;
	y = u.y;
}

struct vec4
{
	vec4() { }
	vec4(sd_scalar x, sd_scalar y, sd_scalar z, sd_scalar w) : x(x), y(y), z(z), w(w) { }
	vec4(const sd_scalar* xyzw) : x(xyzw[0]), y(xyzw[1]), z(xyzw[2]), w(xyzw[3]) { }
	vec4(const vec3& u) : x(u.x), y(u.y), z(u.z), w(1.0f) { }
	vec4(const vec4& u) : x(u.x), y(u.y), z(u.z), w(u.w) { }

	bool operator==(const vec4 & u) const
	{
		return (u.x == x && u.y == y && u.z == z && u.w == w) ? true : false;
	}

	bool operator!=( const vec4& rhs ) const
	{
		return !(*this == rhs );
	}


	vec4 & operator*=(const sd_scalar & lambda)
	{
		x*= lambda;
		y*= lambda;
		z*= lambda;
		w*= lambda;
		return *this;
	}

	vec4 & operator-=(const vec4 & u)
	{
		x-= u.x;
		y-= u.y;
		z-= u.z;
		w-= u.w;
		return *this;
	}

	vec4 & operator+=(const vec4 & u)
	{
		x+= u.x;
		y+= u.y;
		z+= u.z;
		w+= u.w;
		return *this;
	}

	vec4 operator - () const
	{
		return vec4(-x, -y, -z, -w);
	}

	sd_scalar & operator[](int i)
	{
		return vec_array[i];
	}

	const sd_scalar operator[](int i) const
	{
		return vec_array[i];
	}

	union {
		struct {
			sd_scalar x,y,z,w;          // standard names for components
		};
		struct {
			sd_scalar r,g,b,a;          // standard names for components
		};
		struct {
			sd_scalar s,t,r,q;          // standard names for components
		};
		sd_scalar vec_array[4];     // array access
	};
};

inline const vec4 operator+(const vec4& u, const vec4& v)
{
	return vec4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w);
}

inline const vec4 operator-(const vec4& u, const vec4& v)
{
	return vec4(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w);
}

inline const vec4 operator*(const sd_scalar s, const vec4& u)
{
	return vec4(s * u.x, s * u.y, s * u.z, s * u.w);
}

inline const vec4 operator/(const vec4& u, const sd_scalar s)
{
	return vec4(u.x / s, u.y / s, u.z / s, u.w / s);
}

inline const vec4 operator*(const vec4& u, const vec4& v)
{
	return vec4(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

inline vec3::vec3(const vec4& u)
{
	x = u.x;
	y = u.y;
	z = u.z;
}

// quaternion
struct quat;  

/*
for all the matrices...a<x><y> indicates the element at row x, col y

For example:
a01 <-> row 0, col 1 
*/

struct mat3
{
	mat3();
	mat3(const sd_scalar * array);
	mat3(const mat3 & M);
	mat3( const sd_scalar& f0,  const sd_scalar& f1,  const sd_scalar& f2,  
		const sd_scalar& f3,  const sd_scalar& f4,  const sd_scalar& f5,  
		const sd_scalar& f6,  const sd_scalar& f7,  const sd_scalar& f8 )
		: a00( f0 ), a10( f1 ), a20( f2 ), 
		a01( f3 ), a11( f4 ), a21( f5 ),
		a02( f6 ), a12( f7 ), a22( f8) { }

	const vec3 col(const int i) const
	{
		return vec3(&mat_array[i * 3]);
	}

	const vec3 operator[](int i) const
	{
		return vec3(mat_array[i], mat_array[i + 3], mat_array[i + 6]);
	}

	const sd_scalar& operator()(const int& i, const int& j) const
	{
		return mat_array[ j * 3 + i ];
	}

	sd_scalar& operator()(const int& i, const int& j)
	{
		return  mat_array[ j * 3 + i ];
	}

	mat3 & operator*=(const sd_scalar & lambda)
	{
		for (int i = 0; i < 9; ++i)
			mat_array[i] *= lambda;
		return *this;
	}

	mat3 & operator-=(const mat3 & M)
	{
		for (int i = 0; i < 9; ++i)
			mat_array[i] -= M.mat_array[i];
		return *this;
	}

	void set_row(int i, const vec3 & v)
	{
		mat_array[i] = v.x;
		mat_array[i + 3] = v.y;
		mat_array[i + 6] = v.z;
	}

	void set_col(int i, const vec3 & v)
	{
		mat_array[i * 3] = v.x;
		mat_array[i * 3 + 1] = v.y;
		mat_array[i * 3 + 2] = v.z;
	}

	void set_rot(const sd_scalar & theta, const vec3 & v);
	void set_rot(const vec3 & u, const vec3 & v);
	void set_rot(const vec3 & u); // x-y-z
	void get_rot(vec3& u);
	void set_rot_zyx(const vec3 & u); //z-y-x
	void get_rot_zyx(vec3& u);
	void set_rot_zxy(const vec3 & u); //z-x-y
	void get_rot_zxy(vec3& u);
	void set_rot_yxz(const vec3& u); // y-x-z
	void get_rot_yxz(vec3& u); // y-x-z
	void set_rodrigues(const vec3 & u);
	vec3 get_rodrigues();
	void ortho();

	// Matrix norms...
	// Compute || M ||
	//                1
	sd_scalar norm_one();

	// Compute || M ||
	//                +inf
	sd_scalar norm_inf();

	union {
		struct {
			sd_scalar a00, a10, a20;        // standard names for components
			sd_scalar a01, a11, a21;        // standard names for components
			sd_scalar a02, a12, a22;        // standard names for components
		};
		sd_scalar mat_array[9];     // array access
		sd_scalar mat_33[3][3];
	};
};

const vec3 operator*(const mat3&, const vec3&);
const vec3 operator*(const vec3&, const mat3&);

struct mat4
{
	mat4();
	mat4(const sd_scalar * array);
	mat4(const mat4 & M);

	mat4( const sd_scalar& f0,  const sd_scalar& f1,  const sd_scalar& f2,  const sd_scalar& f3,
		const sd_scalar& f4,  const sd_scalar& f5,  const sd_scalar& f6,  const sd_scalar& f7,
		const sd_scalar& f8,  const sd_scalar& f9,  const sd_scalar& f10, const sd_scalar& f11,
		const sd_scalar& f12, const sd_scalar& f13, const sd_scalar& f14, const sd_scalar& f15 )
		: a00( f0 ), a10( f1 ), a20( f2 ), a30( f3 ),
		a01( f4 ), a11( f5 ), a21( f6 ), a31( f7 ),
		a02( f8 ), a12( f9 ), a22( f10), a32( f11),
		a03( f12), a13( f13), a23( f14), a33( f15) { }

	const vec4 col(const int i) const
	{
		return vec4(&mat_array[i * 4]);
	}

	const vec4 operator[](const int& i) const
	{
		return vec4(mat_array[i], mat_array[i + 4], mat_array[i + 8], mat_array[i + 12]);
	}

	const sd_scalar& operator()(const int& i, const int& j) const
	{
		return mat_array[ j * 4 + i ];
	}

	sd_scalar& operator()(const int& i, const int& j)
	{
		return  mat_array[ j * 4 + i ];
	}

	void set_mat3(mat3 & mat);
	const mat3 get_mat3();

	void set_col(int i, const vec4 & v)
	{
		mat_array[i * 4] = v.x;
		mat_array[i * 4 + 1] = v.y;
		mat_array[i * 4 + 2] = v.z;
		mat_array[i * 4 + 3] = v.w;
	}

	void set_row(int i, const vec4 & v)
	{
		mat_array[i] = v.x;
		mat_array[i + 4] = v.y;
		mat_array[i + 8] = v.z;
		mat_array[i + 12] = v.w;
	}

	mat3 & get_rot(mat3 & M) const;
	quat & get_rot(quat & q) const;
	void set_rot(const quat & q);
	void set_rot(const mat3 & M);
	void set_rot(const sd_scalar & theta, const vec3 & v);
	void set_rot(const vec3 & u, const vec3 & v);
	void set_rot(const vec3 & u);
	void get_rot(vec3 & u);

	void set_scale(const vec3& s);
	vec3& get_scale(vec3& s) const;
	void set_translation(const vec3 & t);
	vec3 & get_translation(vec3 & t) const;

	mat4 operator*(const mat4&) const;

	void zeroProcess();

	union {
		struct {
			sd_scalar a00, a10, a20, a30;   // standard names for components
			sd_scalar a01, a11, a21, a31;   // standard names for components
			sd_scalar a02, a12, a22, a32;   // standard names for components
			sd_scalar a03, a13, a23, a33;   // standard names for components
		};
		struct {
			sd_scalar _11, _12, _13, _14;   // standard names for components
			sd_scalar _21, _22, _23, _24;   // standard names for components
			sd_scalar _31, _32, _33, _34;   // standard names for components
			sd_scalar _41, _42, _43, _44;   // standard names for components
		};
		union {
			struct {
				sd_scalar b00, b10, b20, p; // standard names for components
				sd_scalar b01, b11, b21, q; // standard names for components
				sd_scalar b02, b12, b22, r; // standard names for components
				sd_scalar x, y, z, w;       // standard names for components
			};
		};
		sd_scalar mat_array[16];     // array access
		sd_scalar mat_44[4][4];
	};
};

const vec4 operator*(const mat4&, const vec4&);
const vec4 operator*(const vec4&, const mat4&);

// quaternion
struct quat {
public:
	quat();
	quat(sd_scalar x, sd_scalar y, sd_scalar z, sd_scalar w);
	quat(const quat& quat);
	quat(const vec3& axis, sd_scalar angle);
	quat(const mat3& rot);
	quat& operator=(const quat& quat);
	quat operator-()
	{
		return quat(-x, -y, -z, -w);
	}
	quat Inverse();
	void Normalize();
	void FromMatrix(const mat3& mat);
	void ToMatrix(mat3& mat) const;
	quat& operator*=(const quat& q);
	sd_scalar getEulerAngle();
	vec3 GetAxis();
	static const quat Identity;
	sd_scalar& operator[](int i) { return comp[i]; }
	const sd_scalar operator[](int i) const { return comp[i]; }
	union {
		struct {
			sd_scalar x, y, z, w;
		};
		sd_scalar comp[4];
	};
};
const quat operator*(const quat&, const quat&);
extern quat & add_quats(quat & p, const quat & q1, const quat & q2);
extern sd_scalar dot(const quat & p, const quat & q);
extern quat & dot(sd_scalar s, const quat & p, const quat & q);
extern quat & slerp_quats(quat & p, sd_scalar s, const quat & q1, const quat & q2);
extern quat & axis_to_quat(quat & q, const vec3 & a, const sd_scalar phi);
extern mat3 & quat_2_mat(mat3 &M, const quat &q );
extern quat & mat_2_quat(quat &q,const mat3 &M);

struct transformation
{
public:
	vec3 pos;
	vec3 rotv;
};

// constant algebraic values
static const sd_scalar array16_id[] =        { sd_one, sd_zero, sd_zero, sd_zero,
sd_zero, sd_one, sd_zero, sd_zero,
sd_zero, sd_zero, sd_one, sd_zero,
sd_zero, sd_zero, sd_zero, sd_one};

static const sd_scalar array16_null[] =      { sd_zero, sd_zero, sd_zero, sd_zero,
sd_zero, sd_zero, sd_zero, sd_zero,
sd_zero, sd_zero, sd_zero, sd_zero,
sd_zero, sd_zero, sd_zero, sd_zero};

static const sd_scalar array16_scale_bias[] = { sd_zero_5, sd_zero,   sd_zero,   sd_zero,
sd_zero,   sd_zero_5, sd_zero,   sd_zero,
sd_zero,   sd_zero,   sd_zero_5, sd_zero,
sd_zero_5, sd_zero_5, sd_zero_5, sd_one};

static const sd_scalar array9_id[] =         { sd_one, sd_zero, sd_zero,
sd_zero, sd_one, sd_zero,
sd_zero, sd_zero, sd_one};


static const vec2      vec2_null(sd_zero,sd_zero);
static const vec4      vec4_one(sd_one,sd_one,sd_one,sd_one);
static const vec3      vec3_one(sd_one,sd_one,sd_one);
static const vec3      vec3_null(sd_zero,sd_zero,sd_zero);
static const vec3      vec3_x(sd_one,sd_zero,sd_zero);
static const vec3      vec3_y(sd_zero,sd_one,sd_zero);
static const vec3      vec3_z(sd_zero,sd_zero,sd_one);
static const vec3      vec3_neg_x(-sd_one,sd_zero,sd_zero);
static const vec3      vec3_neg_y(sd_zero,-sd_one,sd_zero);
static const vec3      vec3_neg_z(sd_zero,sd_zero,-sd_one);
static const vec4      vec4_null(sd_zero,sd_zero,sd_zero,sd_zero);
static const vec4      vec4_x(sd_one,sd_zero,sd_zero,sd_zero);
static const vec4      vec4_neg_x(-sd_one,sd_zero,sd_zero,sd_zero);
static const vec4      vec4_y(sd_zero,sd_one,sd_zero,sd_zero);
static const vec4      vec4_neg_y(sd_zero,-sd_one,sd_zero,sd_zero);
static const vec4      vec4_z(sd_zero,sd_zero,sd_one,sd_zero);
static const vec4      vec4_neg_z(sd_zero,sd_zero,-sd_one,sd_zero);
static const vec4      vec4_w(sd_zero,sd_zero,sd_zero,sd_one);
static const vec4      vec4_neg_w(sd_zero,sd_zero,sd_zero,-sd_one);

#ifndef EXTERNAL_REFERENCE
static const quat      quat_id(sd_zero,sd_zero,sd_zero,sd_one);
static const mat4      mat4_id(array16_id);
static const mat3      mat3_id(array9_id);
static const mat4      mat4_null(array16_null);
static const mat4      mat4_scale_bias(array16_scale_bias);
#endif //EXTERNAL_REFERENCE

// normalizes a vector and return a reference of itself
extern vec2 & normalize(vec2 & u);
extern vec3 & normalize(vec3 & u);
extern vec4 & normalize(vec4 & u);

// Computes the squared magnitude
inline sd_scalar sd_sq_norm(const vec3 & n)
{ return n.x * n.x + n.y * n.y + n.z * n.z; }

inline sd_scalar sd_sq_norm(const vec4 & n)
{ return n.x * n.x + n.y * n.y + n.z * n.z + n.w * n.w; }

// Computes the magnitude
inline sd_scalar sd_norm(const vec3 & n)
{ return sqrtf(sd_sq_norm(n)); }

inline sd_scalar sd_norm(const vec4 & n)
{ return sqrtf(sd_sq_norm(n)); }


// computes the cross product ( v cross w) and stores the result in u
// i.e.     u = v cross w
extern vec3 & cross(vec3 & u, const vec3 & v, const vec3 & w);

// computes the dot product ( v dot w) and stores the result in u
// i.e.     u = v dot w
extern sd_scalar & dot(sd_scalar & u, const vec3 & v, const vec3 & w);
extern sd_scalar dot(const vec3 & v, const vec3 & w);
extern sd_scalar & dot(sd_scalar & u, const vec4 & v, const vec4 & w);
extern sd_scalar dot(const vec4 & v, const vec4 & w);
extern sd_scalar & dot(sd_scalar & u, const vec3 & v, const vec4 & w);
extern sd_scalar dot(const vec3 & v, const vec4 & w);
extern sd_scalar & dot(sd_scalar & u, const vec4 & v, const vec3 & w);
extern sd_scalar dot(const vec4 & v, const vec3 & w);

// compute the reflected vector R of L w.r.t N - vectors need to be 
// normalized
//
//                R     N     L
//                  _       _
//                 |\   ^   /|
//                   \  |  /
//                    \ | /
//                     \|/
//                      +
extern vec3 & reflect(vec3 & r, const vec3 & n, const vec3 & l);

// Computes u = v * lambda + u
extern vec3 & madd(vec3 & u, const vec3 & v, const sd_scalar & lambda);
// Computes u = v * lambda
extern vec3 & mult(vec3 & u, const vec3 & v, const sd_scalar & lambda);
// Computes u = v * w
extern vec3 & mult(vec3 & u, const vec3 & v, const vec3 & w);
// Computes u = v + w
extern vec3 & add(vec3 & u, const vec3 & v, const vec3 & w);
// Computes u = v - w
extern vec3 & sub(vec3 & u, const vec3 & v, const vec3 & w);

// Computes u = u * s
extern vec2 & scale(vec2 & u, const sd_scalar s);
extern vec3 & scale(vec3 & u, const sd_scalar s);
extern vec4 & scale(vec4 & u, const sd_scalar s);

// Computes u = M * v
extern vec3 & mult(vec3 & u, const mat3 & M, const vec3 & v);
extern vec4 & mult(vec4 & u, const mat4 & M, const vec4 & v);

// Computes u = v * M
extern vec3 & mult(vec3 & u, const vec3 & v, const mat3 & M);
extern vec4 & mult(vec4 & u, const vec4 & v, const mat4 & M);

// Computes u = M(4x4) * v and divides by w
extern vec3 & mult_pos(vec3 & u, const mat4 & M, const vec3 & v);
// Computes u = M(4x4) * v
extern vec3 & mult_dir(vec3 & u, const mat4 & M, const vec3 & v);
// Computes u = M(4x4) * v and does not divide by w (assumed to be 1)
extern vec3 & mult(vec3& u, const mat4& M, const vec3& v);

// Computes u = v * M(4x4) and divides by w
extern vec3 & mult_pos(vec3 & u, const vec3 & v, const mat4 & M);
// Computes u = v * M(4x4)
extern vec3 & mult_dir(vec3 & u, const vec3 & v, const mat4 & M);
// Computes u = v * M(4x4) and does not divide by w (assumed to be 1)
extern vec3 & mult(vec3& u, const vec3& v, const mat4& M);

// Computes A += B
extern mat4 & add(mat4 & A, const mat4 & B);
extern mat3 & add(mat3 & A, const mat3 & B);

// Computes C = A + B
extern mat4 & add(mat4 & C, const mat4 & A, const mat4 & B);
extern mat3 & add(mat3 & C, const mat3 & A, const mat3 & B);

// Computes C = A * B
extern mat4 & mult(mat4 & C, const mat4 & A, const mat4 & B);
extern mat3 & mult(mat3 & C, const mat3 & A, const mat3 & B);
extern mat3 mult(const mat3 & A, const mat3 & B);

// Compute M = -M
extern mat4 & negate(mat4 & M);
extern mat3 & negate(mat3 & M);

// Computes B = Transpose(A)
//       T
//  B = A
extern mat3 & transpose(mat3 & B, const mat3 & A);
extern mat4 & transpose(mat4 & B, const mat4 & A);

// Computes B = Transpose(B)
//       T
//  B = B
extern mat3 & transpose(mat3 & B);
extern mat4 & transpose(mat4 & B);

// Computes B = inverse(A)
//       -1
//  B = A
extern mat4 & invert(mat4 & B, const mat4 & A);
extern mat3 & invert(mat3 & B, const mat3 & A);

// Computes B = inverse(A)
//                                       T  T
//                   (R t)             (R -R t)
// assuming that A = (0 1) so that B = (0    1)
//  B = A
extern mat4 & invert_rot_trans(mat4 & B, const mat4 & A);

extern mat4 & look_at(mat4 & M, const vec3 & eye, const vec3 & center, const vec3 & up);
extern mat4 & frustum(mat4 & M, const sd_scalar l, const sd_scalar r, const sd_scalar b, 
					  const sd_scalar t, const sd_scalar n, const sd_scalar f);

extern mat4 & perspective(mat4 & M, const sd_scalar fovy, const sd_scalar aspect, const sd_scalar n, const sd_scalar f);
extern mat4 & ortho(mat4 & M, const sd_scalar left, 
					const sd_scalar right, 
					const sd_scalar bottom, 
					const sd_scalar top,
					const sd_scalar n,
					const sd_scalar f);

sd_scalar intersects(const vec3 org, const vec3 raydir, const vec3 sphereOrg, sd_scalar radius, bool discardInside);
void getPickRay(int mouseX, int mouseY, mat4 pmatView, mat4 pmatProj, sd_scalar viewPortX, sd_scalar viewPortY, sd_scalar viewPortW, sd_scalar viewPortH, vec3& vPickRayDir, vec3& vPickRayOrig);

/* Decompose Affine Matrix 
*    A = TQS, where
* A is the affine transform
* T is the translation vector
* Q is the rotation (quaternion)
* S is the scale vector
* f is the sign of the determinant
*/
extern void decomp_affine(const mat4 & A, vec3 & T, quat & Q, quat & U, vec3 & S, sd_scalar & f);
// quaternion
extern quat & normalize(quat & p);
extern quat & conj(quat & p);
extern quat & conj(quat & p, const quat & q);
extern quat & add_quats(quat & p, const quat & q1, const quat & q2);
extern quat & axis_to_quat(quat & q, const vec3 & a, const sd_scalar phi);
extern mat3 & quat_2_mat(mat3 &M, const quat &q );
extern quat & mat_2_quat(quat &q,const mat3 &M);
extern quat & mat_2_quat(quat &q,const mat4 &M);

// surface properties
extern mat3 & tangent_basis(mat3 & basis,const vec3 & v0,const vec3 & v1,const vec3 & v2,const vec2 & t0,const vec2 & t1,const vec2 & t2, const vec3 & n);

// linear interpolation
inline sd_scalar lerp(sd_scalar t, sd_scalar a, sd_scalar b)
{ return a * (sd_one - t) + t * b; }

inline vec3 & lerp(vec3 & w, const sd_scalar & t, const vec3 & u, const vec3 & v)
{ w.x = lerp(t, u.x, v.x); w.y = lerp(t, u.y, v.y); w.z = lerp(t, u.z, v.z); return w; }

inline vec4 & lerp(vec4 & w, const sd_scalar & t, const vec4 & u, const vec4 & v)
{ w.x = lerp(t, u.x, v.x); w.y = lerp(t, u.y, v.y); w.z = lerp(t, u.z, v.z); w.w = lerp(t, u.w, v.w); return w; }

// utilities
inline sd_scalar sd_min(const sd_scalar & lambda, const sd_scalar & n)
{ return (lambda < n ) ? lambda : n; }

inline sd_scalar sd_max(const sd_scalar & lambda, const sd_scalar & n)
{ return (lambda > n ) ? lambda : n; }

inline sd_scalar sd_clamp(sd_scalar u, const sd_scalar min, const sd_scalar max)
{ u = (u < min) ? min : u; u = (u > max) ? max : u; return u; }

extern sd_scalar sd_random();

extern quat & trackball(quat & q, vec2 & pt1, vec2 & pt2, sd_scalar trackballsize);

extern vec3 & cube_map_normal(int i, int x, int y, int cubesize, vec3 & v);

// Componentwise maximum and minium 
inline void sd_max(vec3 & vOut, const vec3 & vFirst, const vec3 & vSecond)
{
	vOut.x = sd_max(vFirst.x, vSecond.x);
	vOut.y = sd_max(vFirst.y, vSecond.y);
	vOut.z = sd_max(vFirst.z, vSecond.z);
}

inline void sd_min(vec3 & vOut, const vec3 & vFirst, const vec3 & vSecond)
{
	vOut.x = sd_min(vFirst.x, vSecond.x);
	vOut.y = sd_min(vFirst.y, vSecond.y);
	vOut.z = sd_min(vFirst.z, vSecond.z);
}


// geometry
// computes the area of a triangle
extern sd_scalar sd_area(const vec3 & v1, const vec3 & v2, const vec3 &v3);
// computes the perimeter of a triangle
extern sd_scalar sd_perimeter(const vec3 & v1, const vec3 & v2, const vec3 &v3);
// find the inscribed circle
extern sd_scalar sd_find_in_circle( vec3 & center, const vec3 & v1, const vec3 & v2, const vec3 &v3);
// find the circumscribed circle
extern sd_scalar sd_find_circ_circle( vec3 & center, const vec3 & v1, const vec3 & v2, const vec3 &v3);

// fast cosine functions
extern sd_scalar fast_cos(const sd_scalar x);
extern sd_scalar ffast_cos(const sd_scalar x);

// determinant
sd_scalar det(const mat3 & A);

extern void sd_is_valid(const vec3& v);
extern void sd_is_valid(sd_scalar lambda);

//rotation
extern void get_rot(const vec3& i, const vec3& u, const vec3& v, sd_scalar& theta, vec3& axis);
float interp(float start, float end, float current, int type, bool checkBound = true );

mat3 getRot(vec3 preLeft, vec3 preUp, vec3 afterLeft, vec3 afterUp); // calc rotation that convert (pre L,U) to (post L,U)

//--------------------------------------------------------------------------------------
// Given a ray origin (orig) and direction (dir), and three vertices of a triangle, this
// function returns true and the interpolated texture coordinates if the ray intersects 
// the triangle
//--------------------------------------------------------------------------------------
bool IntersectTriangle( const vec3 orig, const vec3 dir, //ray
	vec3 v0, vec3 v1, vec3 v2, //triangle
	float* t, //Ray-intersection parameter distance.
	float* u, //Barycentric hit coordinates, U.
	float* v );//Barycentric hit coordinates, V.

void dropA2BWithLen(vec3 &A, vec3 B, sd_scalar len);
void clip_limit(sd_scalar &val, sd_scalar min, sd_scalar max);

#endif //_sd_3DMATH_H_
