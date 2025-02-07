#include <algorithm>
#include <cmath>
#include <concepts>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tensorlib {
	template<typename T>
	concept Arithmetic = std::is_arithmetic_v<T>;

	template<Arithmetic T, size_t N>
	class Tensor {
	private:
		std::shared_ptr<std::vector<T>> buffer_;
		std::array<size_t, N> shape_;
		std::array<size_t, N> strides_;
		size_t offset_;

		struct BroadcastMeta {
			std::array<size_t, N> virtual_shape;
			std::array<size_t, N> virtual_strides;
			bool is_broadcasted;
		};

		void compute_strides() noexcept {
			strides_[N - 1] = 1;
			for (int i = N - 2; i >= 0; --i) {
				strides_[i] = strides_[i + 1] * shape_[i + 1];
			}
		}

		template<size_t D>
		size_t flat_index(const std::array<size_t, D>& indices) const {
			static_assert(D == N, "Index dimension mismatch");
			size_t index = offset_;
			for (size_t i = 0; i < N; ++i) {
				if (indices[i] >= shape_[i]) {
					throw std::out_of_range("Tensor index out of bounds");
				}
				index += indices[i] * strides_[i];
			}
			return index;
		}

		BroadcastMeta broadcast_compatibility(const Tensor& other) const {
			BroadcastMeta meta{};
			meta.is_broadcasted = false;

			const auto& a_shape = shape_;
			const auto& b_shape = other.shape_;

			for (int i = N - 1, j = other.rank() - 1; i >= 0 || j >= 0; --i, --j) {
				const size_t dim_a = (i >= 0) ? a_shape[i] : 1;
				const size_t dim_b = (j >= 0) ? b_shape[j] : 1;

				if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
					throw std::invalid_argument("Incompatible shapes for broadcasting");
				}

				const size_t max_dim = std::max(dim_a, dim_b);
				meta.virtual_shape[std::max(i, j)] = max_dim;

				if (dim_a != max_dim || dim_b != max_dim) {
					meta.is_broadcasted = true;
				}
			}

			size_t stride = 1;
			for (int i = N - 1; i >= 0; --i) {
				meta.virtual_strides[i] = (shape_[i] == meta.virtual_shape[i]) ? strides_[i] : 0;
				stride *= meta.virtual_shape[i];
			}

			return meta;
		}

	public:
		using value_type = T;
		static constexpr size_t rank = N;

		Tensor() : shape_({ 0 }), strides_({ 0 }), offset_(0),
			buffer_(std::make_shared<std::vector<T>>()) {
		}

		explicit Tensor(const std::array<size_t, N>& shape) : shape_(shape), offset_(0) {
			size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
			buffer_ = std::make_shared<std::vector<T>>(total_size);
			compute_strides();
		}

		Tensor(const std::array<size_t, N>& shape, const T& init_value) : Tensor(shape) {
			std::fill(buffer_->begin(), buffer_->end(), init_value);
		}

		Tensor(const Tensor&) = default;
		Tensor(Tensor&&) = default;

		template<typename... Args>
			requires (sizeof...(Args) == N) && (std::convertible_to<Args, size_t> && ...)
		T& operator()(Args... args) {
			std::array<size_t, N> indices{ static_cast<size_t>(args)... };
			return (*buffer_)[flat_index(indices)];
		}

		template<typename... Args>
			requires (sizeof...(Args) == N) && (std::convertible_to<Args, size_t> && ...)
		const T& operator()(Args... args) const {
			std::array<size_t, N> indices{ static_cast<size_t>(args)... };
			return (*buffer_)[flat_index(indices)];
		}

		Tensor& operator=(const Tensor& other) {
			if (&other != this) {
				buffer_ = other.buffer_;
				shape_ = other.shape_;
				strides_ = other.strides_;
				offset_ = other.offset_;
			}
			return *this;
		}

		Tensor operator+(const Tensor& other) const {
			auto meta = broadcast_compatibility(other);
			Tensor result(meta.virtual_shape);

			auto it1 = begin();
			auto it2 = other.begin();
			auto rit = result.begin();

			while (rit != result.end()) {
				*rit = *it1 + *it2;
				++rit; ++it1; ++it2;
			}
			return result;
		}

		Tensor operator*(const Tensor& other) const {
			auto meta = broadcast_compatibility(other);
			Tensor result(meta.virtual_shape);

			auto it1 = begin();
			auto it2 = other.begin();
			auto rit = result.begin();

			while (rit != result.end()) {
				*rit = *it1 * *it2;
				++rit; ++it1; ++it2;
			}
			return result;
		}

		Tensor slice(const std::array<size_t, N>& start, const std::array<size_t, N>& end) const {
			Tensor sliced;
			sliced.buffer_ = buffer_;
			sliced.offset_ = offset_;

			for (size_t i = 0; i < N; ++i) {
				if (end[i] <= start[i] || end[i] > shape_[i]) {
					throw std::invalid_argument("Invalid slice range");
				}
				sliced.shape_[i] = end[i] - start[i];
				sliced.offset_ += start[i] * strides_[i];
			}
			sliced.compute_strides();
			return sliced;
		}

		Tensor transpose() const requires (N == 2) {
			Tensor transposed({ shape_[1], shape_[0] });
			for (size_t i = 0; i < shape_[0]; ++i) {
				for (size_t j = 0; j < shape_[1]; ++j) {
					transposed(j, i) = (*this)(i, j);
				}
			}
			return transposed;
		}

		template<Arithmetic U>
		Tensor<U, N> astype() const {
			Tensor<U, N> converted(shape_);
			std::transform(begin(), end(), converted.begin(),
				[](const T& val) { return static_cast<U>(val); });
			return converted;
		}

		class iterator {
		public:
			using iterator_category = std::random_access_iterator_tag;
			using value_type = T;
			using difference_type = std::ptrdiff_t;
			using pointer = T*;
			using reference = T&;

			iterator(Tensor& tensor, size_t pos = 0)
				: data_(tensor.buffer_->data()), strides_(tensor.strides_),
				shape_(tensor.shape_), offset_(tensor.offset_), pos_(pos) {
			}

			reference operator*() { return data_[offset_ + pos_]; }
			pointer operator->() { return data_ + offset_ + pos_; }

			iterator& operator++() { ++pos_; return *this; }
			iterator operator++(int) { iterator tmp = *this; ++pos_; return tmp; }

			bool operator==(const iterator& other) const { return pos_ == other.pos_; }
			bool operator!=(const iterator& other) const { return pos_ != other.pos_; }

		private:
			T* data_;
			std::array<size_t, N> strides_;
			std::array<size_t, N> shape_;
			size_t offset_;
			size_t pos_;
		};

		iterator begin() { return iterator(*this, 0); }
		iterator end() { return iterator(*this, buffer_->size()); }

		class const_iterator {
		public:
			using iterator_category = std::random_access_iterator_tag;
			using value_type = const T;
			using difference_type = std::ptrdiff_t;
			using pointer = const T*;
			using reference = const T&;

			const_iterator(const Tensor& tensor, size_t pos = 0)
				: data_(tensor.buffer_->data()), strides_(tensor.strides_),
				shape_(tensor.shape_), offset_(tensor.offset_), pos_(pos) {
			}

			reference operator*() { return data_[offset_ + pos_]; }
			pointer operator->() { return data_ + offset_ + pos_; }

			const_iterator& operator++() { ++pos_; return *this; }
			const_iterator operator++(int) { const_iterator tmp = *this; ++pos_; return tmp; }

			bool operator==(const const_iterator& other) const { return pos_ == other.pos_; }
			bool operator!=(const const_iterator& other) const { return pos_ != other.pos_; }

		private:
			const T* data_;
			std::array<size_t, N> strides_;
			std::array<size_t, N> shape_;
			size_t offset_;
			size_t pos_;
		};

		const_iterator begin() const { return const_iterator(*this, 0); }
		const_iterator end() const { return const_iterator(*this, buffer_->size()); }

		Tensor& apply_function(const std::function<T(T)>& fn) {
			std::transform(begin(), end(), begin(), fn);
			return *this;
		}

		Tensor sqrt() const {
			Tensor result(*this);
			return result.apply_function([](T x) { return std::sqrt(x); });
		}

		Tensor exp() const {
			Tensor result(*this);
			return result.apply_function([](T x) { return std::exp(x); });
		}

		Tensor log() const {
			Tensor result(*this);
			return result.apply_function([](T x) { return std::log(x); });
		}

		T sum() const {
			return std::accumulate(begin(), end(), T(0));
		}

		T max() const {
			return *std::max_element(begin(), end());
		}

		T min() const {
			return *std::min_element(begin(), end());
		}

		Tensor reshape(const std::array<size_t, N>& new_shape) const {
			size_t total_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
			size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());

			if (total_size != new_total) {
				throw std::invalid_argument("Total elements must match for reshape");
			}

			Tensor reshaped = *this;
			reshaped.shape_ = new_shape;
			reshaped.compute_strides();
			return reshaped;
		}

		bool contiguous() const noexcept {
			size_t stride = 1;
			for (int i = N - 1; i >= 0; --i) {
				if (strides_[i] != stride) return false;
				stride *= shape_[i];
			}
			return true;
		}

		const std::array<size_t, N>& shape() const noexcept { return shape_; }
		size_t size() const noexcept { return buffer_->size(); }
	};

	template<Arithmetic T, size_t N, size_t M>
	Tensor<T, N + M> outer_product(const Tensor<T, N>& a, const Tensor<T, M>& b) {
		std::array<size_t, N + M> new_shape;
		std::copy(a.shape().begin(), a.shape().end(), new_shape.begin());
		std::copy(b.shape().begin(), b.shape().end(), new_shape.begin() + N);

		Tensor<T, N + M> result(new_shape);

		auto it = result.begin();
		for (const auto& aval : a) {
			for (const auto& bval : b) {
				*it++ = aval * bval;
			}
		}
		return result;
	}

	template<Arithmetic T, size_t N>
	Tensor<T, N> linspace(T start, T stop, const std::array<size_t, N>& shape) {
		Tensor<T, N> tensor(shape);
		T step = (stop - start) / (tensor.size() - 1);
		T value = start;
		for (auto& elem : tensor) {
			elem = value;
			value += step;
		}
		return tensor;
	}

	template<Arithmetic T, size_t N>
	Tensor<T, N> zeros(const std::array<size_t, N>& shape) {
		return Tensor<T, N>(shape, T(0));
	}

	template<Arithmetic T, size_t N>
	Tensor<T, N> ones(const std::array<size_t, N>& shape) {
		return Tensor<T, N>(shape, T(1));
	}
} // namespace tensorlib
