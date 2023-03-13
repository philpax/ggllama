use std::{ffi::c_void, ptr::NonNull};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GgmlError {
    #[error("integer conversion error")]
    TryFromInt(#[from] std::num::TryFromIntError),
    #[error("failed to create resource")]
    ResourceCreationFailure,
}

pub type Result<T> = core::result::Result<T, GgmlError>;

pub trait SysType {
    type Sys;
    unsafe fn as_ptr(&self) -> *const Self::Sys;
    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Sys;
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(not(windows), repr(u32))]
#[cfg_attr(windows, repr(i32))]
pub enum Type {
    Q4_0 = ggml_sys::ggml_type_GGML_TYPE_Q4_0,
    Q4_1 = ggml_sys::ggml_type_GGML_TYPE_Q4_1,
    I8 = ggml_sys::ggml_type_GGML_TYPE_I8,
    I16 = ggml_sys::ggml_type_GGML_TYPE_I16,
    I32 = ggml_sys::ggml_type_GGML_TYPE_I32,
    F16 = ggml_sys::ggml_type_GGML_TYPE_F16,
    F32 = ggml_sys::ggml_type_GGML_TYPE_F32,
    COUNT = ggml_sys::ggml_type_GGML_TYPE_COUNT,
}
impl Type {
    pub fn to_sys(&self) -> ggml_sys::ggml_type {
        *self as ggml_sys::ggml_type
    }
    pub fn from_sys(sys: ggml_sys::ggml_type) -> Self {
        match sys {
            ggml_sys::ggml_type_GGML_TYPE_Q4_0 => Self::Q4_0,
            ggml_sys::ggml_type_GGML_TYPE_Q4_1 => Self::Q4_1,
            ggml_sys::ggml_type_GGML_TYPE_I8 => Self::I8,
            ggml_sys::ggml_type_GGML_TYPE_I16 => Self::I16,
            ggml_sys::ggml_type_GGML_TYPE_I32 => Self::I32,
            ggml_sys::ggml_type_GGML_TYPE_F16 => Self::F16,
            ggml_sys::ggml_type_GGML_TYPE_F32 => Self::F32,
            ggml_sys::ggml_type_GGML_TYPE_COUNT => Self::COUNT,
            _ => unreachable!(),
        }
    }

    pub fn size(&self) -> usize {
        unsafe { ggml_sys::ggml_type_size(self.to_sys()) }
    }

    pub fn blck_size(&self) -> Result<usize> {
        Ok(unsafe { ggml_sys::ggml_blck_size(self.to_sys()) }.try_into()?)
    }

    /// [Self::size]/[Self::blck_size] as float
    pub fn sizef(&self) -> Result<f32> {
        Ok(unsafe { ggml_sys::ggml_type_sizef(self.to_sys()) })
    }
}

// TODO: Make this borrow from the Context so that it's impossible to keep a Tensor
// past the lifetime of its parent Context. This can be unsound.
#[derive(Copy, Clone)]
pub struct Tensor(NonNull<ggml_sys::ggml_tensor>);

impl SysType for Tensor {
    type Sys = ggml_sys::ggml_tensor;
    unsafe fn as_ptr(&self) -> *const Self::Sys {
        self.0.as_ptr()
    }
    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Sys {
        self.0.as_ptr()
    }
}
impl Tensor {
    pub fn ne(&self) -> &[i32] {
        &unsafe { self.0.as_ref() }.ne
    }
    pub fn nb(&self) -> &[usize] {
        &unsafe { self.0.as_ref() }.nb
    }

    pub fn element_size(&self) -> usize {
        unsafe { ggml_sys::ggml_element_size(self.as_ptr()) }
    }

    pub fn n_bytes(&self) -> usize {
        unsafe { ggml_sys::ggml_nbytes(self.as_ptr()) }
    }

    pub fn n_elements(&self) -> Result<usize> {
        Ok(unsafe { ggml_sys::ggml_nelements(self.as_ptr()) }.try_into()?)
    }
    pub fn type_(&self) -> Type {
        Type::from_sys(unsafe { self.0.as_ref().type_ })
    }

    pub fn as_mut_slice<T: TensorType>(&mut self) -> &mut [T] {
        let value_type = T::value_type();
        assert_eq!(self.type_(), value_type);
        unsafe {
            std::slice::from_raw_parts_mut(
                self.0.as_mut().data as *mut T,
                self.n_bytes() / value_type.size(),
            )
        }
    }

    pub fn as_mut_slice_u8(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.0.as_mut().data as *mut u8, self.n_bytes()) }
    }
}

pub trait TensorType {
    fn value_type() -> Type;
}

impl TensorType for i8 {
    fn value_type() -> Type {
        Type::I8
    }
}
impl TensorType for i16 {
    fn value_type() -> Type {
        Type::I16
    }
}
impl TensorType for i32 {
    fn value_type() -> Type {
        Type::I32
    }
}
impl TensorType for f32 {
    fn value_type() -> Type {
        Type::F32
    }
}

pub struct Context(NonNull<ggml_sys::ggml_context>);
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ggml_sys::ggml_free(self.as_mut_ptr()) }
    }
}
impl SysType for Context {
    type Sys = ggml_sys::ggml_context;
    unsafe fn as_ptr(&self) -> *const Self::Sys {
        self.0.as_ptr()
    }
    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Sys {
        self.0.as_ptr()
    }
}
impl Context {
    pub fn new(mem_size: usize, mem_buffer: Option<NonNull<u8>>) -> Option<Self> {
        Some(Self(NonNull::new(unsafe {
            ggml_sys::ggml_init(ggml_sys::ggml_init_params {
                mem_size,
                mem_buffer: mem_buffer
                    .map(|b| b.as_ptr() as *mut c_void)
                    .unwrap_or(std::ptr::null_mut()),
            })
        })?))
    }

    pub fn new_tensor_f32(&mut self, value: f32) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe { ggml_sys::ggml_new_f32(self.as_mut_ptr(), value) })
    }

    pub fn new_tensor_1d(&mut self, type_: Type, ne0: usize) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_new_tensor_1d(self.as_mut_ptr(), type_.to_sys(), ne0.try_into()?)
        })
    }

    pub fn new_tensor_2d(&mut self, type_: Type, ne0: usize, ne1: usize) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_new_tensor_2d(
                self.as_mut_ptr(),
                type_.to_sys(),
                ne0.try_into()?,
                ne1.try_into()?,
            )
        })
    }

    pub fn new_tensor_3d(
        &mut self,
        type_: Type,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_new_tensor_3d(
                self.as_mut_ptr(),
                type_.to_sys(),
                ne0.try_into()?,
                ne1.try_into()?,
                ne2.try_into()?,
            )
        })
    }

    pub fn add(&mut self, mut a: Tensor, mut b: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_add(self.as_mut_ptr(), a.as_mut_ptr(), b.as_mut_ptr())
        })
    }

    pub fn cpy(&mut self, mut a: Tensor, mut b: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_cpy(self.as_mut_ptr(), a.as_mut_ptr(), b.as_mut_ptr())
        })
    }

    pub fn diag_mask_inf(&mut self, mut a: Tensor, n_past: usize) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_diag_mask_inf(self.as_mut_ptr(), a.as_mut_ptr(), n_past.try_into()?)
        })
    }

    pub fn get_rows(&mut self, mut a: Tensor, mut b: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_get_rows(self.as_mut_ptr(), a.as_mut_ptr(), b.as_mut_ptr())
        })
    }

    pub fn mul(&mut self, mut a: Tensor, mut b: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_mul(self.as_mut_ptr(), a.as_mut_ptr(), b.as_mut_ptr())
        })
    }

    pub fn mul_mat(&mut self, mut a: Tensor, mut b: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_mul_mat(self.as_mut_ptr(), a.as_mut_ptr(), b.as_mut_ptr())
        })
    }

    pub fn norm(&mut self, mut a: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_norm(self.as_mut_ptr(), a.as_mut_ptr())
        })
    }

    pub fn permute(
        &mut self,
        mut a: Tensor,
        axis0: usize,
        axis1: usize,
        axis2: usize,
        axis3: usize,
    ) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_permute(
                self.as_mut_ptr(),
                a.as_mut_ptr(),
                axis0.try_into()?,
                axis1.try_into()?,
                axis2.try_into()?,
                axis3.try_into()?,
            )
        })
    }

    pub fn repeat(&mut self, mut a: Tensor, mut b: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_repeat(self.as_mut_ptr(), a.as_mut_ptr(), b.as_mut_ptr())
        })
    }

    pub fn reshape_3d(
        &mut self,
        mut a: Tensor,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_reshape_3d(
                self.as_mut_ptr(),
                a.as_mut_ptr(),
                ne0.try_into()?,
                ne1.try_into()?,
                ne2.try_into()?,
            )
        })
    }

    pub fn rope(
        &mut self,
        mut a: Tensor,
        n_past: usize,
        n_dims: usize,
        mode: i32,
    ) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_rope(
                self.as_mut_ptr(),
                a.as_mut_ptr(),
                n_past.try_into()?,
                n_dims.try_into()?,
                mode,
            )
        })
    }

    pub fn scale(&mut self, mut a: Tensor, mut b: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_scale(self.as_mut_ptr(), a.as_mut_ptr(), b.as_mut_ptr())
        })
    }

    pub fn silu(&mut self, mut a: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_silu(self.as_mut_ptr(), a.as_mut_ptr())
        })
    }

    pub fn soft_max(&mut self, mut a: Tensor) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_soft_max(self.as_mut_ptr(), a.as_mut_ptr())
        })
    }

    pub fn used_memory(&self) -> usize {
        unsafe { ggml_sys::ggml_used_mem(self.as_ptr()) }
    }

    pub fn view_1d(&mut self, mut a: Tensor, ne0: usize, offset: usize) -> Result<Tensor> {
        Self::make_tensor_from_ptr(unsafe {
            ggml_sys::ggml_view_1d(self.as_mut_ptr(), a.as_mut_ptr(), ne0.try_into()?, offset)
        })
    }

    pub fn compute(&mut self, graph: &mut ComputationGraph) {
        unsafe { ggml_sys::ggml_graph_compute(self.as_mut_ptr(), graph.as_mut_ptr()) };
    }
}
impl Context {
    fn make_tensor_from_ptr(ptr: *mut ggml_sys::ggml_tensor) -> Result<Tensor> {
        NonNull::new(ptr)
            .map(|p| Tensor(p))
            .ok_or(GgmlError::ResourceCreationFailure)
    }
}

pub struct ComputationGraph(ggml_sys::ggml_cgraph);
impl ComputationGraph {
    pub fn new(n_threads: usize) -> Result<Self> {
        let mut cgraph: ggml_sys::ggml_cgraph = unsafe { std::mem::zeroed() };
        cgraph.n_threads = n_threads.try_into()?;

        Ok(Self(cgraph))
    }

    pub fn build_forward_expand(&mut self, mut tensor: Tensor) {
        unsafe { ggml_sys::ggml_build_forward_expand(self.as_mut_ptr(), tensor.as_mut_ptr()) };
    }
}
impl SysType for ComputationGraph {
    type Sys = ggml_sys::ggml_cgraph;
    unsafe fn as_ptr(&self) -> *const Self::Sys {
        &self.0
    }
    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Sys {
        &mut self.0
    }
}

pub mod time {
    pub fn init() {
        unsafe { ggml_sys::ggml_time_init() }
    }

    pub fn us() -> i64 {
        unsafe { ggml_sys::ggml_time_us() }
    }
}
