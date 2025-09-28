#pragma once


#include <ceres/ceres.h>
#include <sophus/ceres/manifold.h>
#include <sophus/interp/spline/bspline.h>
#include <sophus/interp/spline/details/bspline_segment.h>

namespace sophus {

    template <typename Scalar_,int kDim>
        struct BSplineErrorSupport {
            typedef enum {
                SPLINE_P0=0,
                SPLINE_P1=1,
                SPLINE_P2=2,
                SPLINE_P3=3,
                SPLINE_Q0=4,
                SPLINE_Q1=5,
                SPLINE_Q2=6,
                SPLINE_Q3=7,
            } SplineParameterId;

            template <typename T>
                using BSpline = CubicBSpline<T,kDim>;
            template <typename T>
                using Vector = typename BSpline<T>::Vector;
            using Vectord = Vector<Scalar_>;
            using BSplined = BSpline<Scalar_>;
            using SegmentCase = sophus::SegmentCase;
            static int constexpr num_parameters = kDim;

            template <class ErrorFunctor, int num_residuals_> 
                struct SplineErrorWrapper0 {
                    static int constexpr num_residuals = num_residuals_;
                    // using SplineErrorWrapper<ErrorFunctor>::call;
                    SplineErrorWrapper0(SegmentCase scase, double u, 
                            std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        pmap(pmap), segment_case(scase), u(u), soph(soph) {
                        }

                    void check_map(unsigned char maxval) const {
                        for (auto x : this->pmap) {
                            SOPHUS_ASSERT(x <= maxval, "but %", x);
                        }
                    }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2};
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(3);
                            return this->template call<T, T>(P[pmap[SPLINE_P0]],
                                    P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],
                                    P[pmap[SPLINE_P3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3};
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(4);
                            return this->template call<T, T>(P[pmap[SPLINE_P0]],
                                    P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],
                                    P[pmap[SPLINE_P3]],residuals);
                        }


                    template <typename T2, typename T>
                        bool call(const T2* const P0, const T2* const P1, const T2* const P2, const T2* const P3, 
                                T* residuals) const {

                            using Map = Eigen::Map<Eigen::Matrix<T2,kDim,1> const>;
                            Map p0(P0), p1(P1), p2(P2), p3(P3);
                            
                            sophus::details::CubicBSplineSegment<T2,kDim> s(this->segment_case,p0,p1,p2,p3);
                            Vector<T> Tt = s.interpolate(this->u).template cast<T>();
                            return this->soph->operator()(Tt.data(),residuals);
                        }

                    std::vector<unsigned char> pmap;
                    SegmentCase segment_case;
                    double u;
                    std::shared_ptr<ErrorFunctor> soph;

                };

            template <class ErrorFunctor,int num_residuals_> 
                struct SplineError2FunctionsWrapper0 {
                    static int constexpr num_residuals = num_residuals_;
                    SplineError2FunctionsWrapper0(SegmentCase scase1, double u1, SegmentCase scase2, double u2, 
                            std::shared_ptr<ErrorFunctor> soph,
                            const std::vector<unsigned char> & pmap) : 
                        pmap(pmap),segment_case1(scase1),segment_case2(scase2), u1(u1), u2(u2), soph(soph) {
                        }


                    void check_map(unsigned char maxval) const {
                        for (auto x : pmap) {
                            SOPHUS_ASSERT(x <= maxval, "but %", x);
                        }
                    }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5};
                            this->check_map(6);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5, const T* P6,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5,P6};
                            this->check_map(7);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5, const T* P6, const T* P7,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5,P6,P7};
                            this->check_map(8);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }


                    template <typename T1, typename T2, typename T>
                        bool call(const T1* const P0, const T1* const P1, const T1* const P2, const T1* const P3, 
                                const T2* const Q0, const T2* const Q1, const T2* const Q2, const T2* const Q3, 
                                T* residuals) const {

                            using Mapper1 = Eigen::Map<Eigen::Matrix<T1,kDim,1>const>;
                            using Mapper2 = Eigen::Map<Eigen::Matrix<T2,kDim,1>const>;
                            Mapper1 p0(P0), p1(P1), p2(P2), p3(P3);
                            Mapper2 q0(Q0), q1(Q1), q2(Q2), q3(Q3);
                            
                            sophus::details::CubicBSplineSegment<T1,kDim> s1(this->segment_case,p0,p1,p2,p3);
                            sophus::details::CubicBSplineSegment<T2,kDim> s2(this->segment_case,q0,q1,q2,q3);
                            Vector<T> t1 = s1.interpolate(this->u).template cast<T>();
                            Vector<T> t2 = s2.interpolate(this->u).template cast<T>();
                            return this->soph->operator()(t1.data(),t2.data(),residuals);
                        }

                    std::vector<unsigned char> pmap;
                    SegmentCase segment_case1, segment_case2;
                    double u1, u2;
                    std::shared_ptr<ErrorFunctor> soph;

                };

            template <class ErrorFunctor,int num_residuals_> 
                struct SplineError2PointsWrapper0 {
                    static int constexpr num_residuals = num_residuals_;
                    // using SplineErrorWrapper<ErrorFunctor>::call;
                    SplineError2PointsWrapper0(SegmentCase scase1, double u1, SegmentCase scase2, double u2, 
                            std::shared_ptr<ErrorFunctor> soph,
                            const std::vector<unsigned char> & pmap) : 
                        pmap(pmap),segment_case1(scase1),segment_case2(scase2), 
                        u1(u1), u2(u2), soph(soph) {
                        }

                    void check_map(unsigned char maxval) const {
                        for (auto x : pmap) {
                            SOPHUS_ASSERT(x <= maxval, "but %", x);
                        }
                    }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2};
                            this->check_map(3);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3};
                            this->check_map(4);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4};
                            this->check_map(5);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5};
                            this->check_map(6);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5, const T* P6,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5,P6};
                            this->check_map(7);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* P0, const T* P1, const T* P2, const T* P3, const T* P4, const T* P5, const T* P6, const T* P7,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3,P4,P5,P6,P7};
                            this->check_map(8);
                            //alias
                            const std::vector<unsigned char> & pmap = this->pmap;
                            return this->template call<T, T, T>(P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],
                                    P[pmap[SPLINE_Q0]],P[pmap[SPLINE_Q1]],P[pmap[SPLINE_Q2]],P[pmap[SPLINE_Q3]],residuals);
                        }


                    template <typename T1, typename T2, typename T>
                        bool call(const T1* const P0, const T1* const P1, const T1* const P2, const T1* const P3, 
                                const T2* const Q0, const T2* const Q1, const T2* const Q2, const T2* const Q3, 
                                T* residuals) const {
                            using Mapper1 = Eigen::Map<Eigen::Matrix<T1,kDim,1>const>;
                            using Mapper2 = Eigen::Map<Eigen::Matrix<T2,kDim,1>const>;
                            Mapper1 p0(P0), p1(P1), p2(P2), p3(P3);
                            Mapper2 q0(Q0), q1(Q1), q2(Q2), q3(Q3);
                            
                            sophus::details::CubicBSplineSegment<T1,kDim> s1(this->segment_case,p0,p1,p2,p3);
                            sophus::details::CubicBSplineSegment<T2,kDim> s2(this->segment_case,q0,q1,q2,q3);
                            Vector<T> t1 = s1.interpolate(this->u).template cast<T>();
                            Vector<T> t2 = s2.interpolate(this->u).template cast<T>();
                            return this->soph->operator()(t1.data(),t2.data(),residuals);

                        }

                    std::vector<unsigned char> pmap;
                    SegmentCase segment_case1, segment_case2;
                    double u1, u2;
                    std::shared_ptr<ErrorFunctor> soph;

                };

            template <class ErrorFunctor,int num_residuals_> 
                struct SplineErrorWrapper1 : public SplineErrorWrapper0<ErrorFunctor,num_residuals_> {
                    SplineErrorWrapper1(SegmentCase scase, double u, std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        SplineErrorWrapper0<ErrorFunctor,num_residuals_>(scase,u,soph,pmap) {}

                    template <typename T>
                        bool operator()(const T* const C0,
                                const T* P0, const T* P1, const T* P2, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2};
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(3);
                            return this->template call<T, T, T>(C0, P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* const C0,
                                const T* P0, const T* P1, const T* P2, const T* P3,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3};
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(4);
                            return this->template call<T, T, T>(C0, P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],residuals);
                        }


                    template <typename T1, typename T2, typename T>
                        bool call(const T1* const C0,
                                const T2* const P0, const T2* const P1, const T2* const P2, const T2* const P3, 
                                T* residuals) const {

                            using Map = Eigen::Map<Eigen::Matrix<T2,kDim,1> const>;
                            Map p0(P0), p1(P1), p2(P2), p3(P3);
                            
                            sophus::details::CubicBSplineSegment<T2,kDim> s(this->segment_case,p0,p1,p2,p3);
                            Vector<T> Tt = s.interpolate(this->u).template cast<T>();
                            return this->soph->operator()(C0,Tt.data(),residuals);
                        }

                };


            template <class ErrorFunctor,int num_residuals_> 
                struct SplineErrorWrapper2 : public SplineErrorWrapper0<ErrorFunctor,num_residuals_> {

                    SplineErrorWrapper2(SegmentCase scase, double u, std::shared_ptr<ErrorFunctor> soph, const std::vector<unsigned char> & pmap) : 
                        SplineErrorWrapper0<ErrorFunctor,num_residuals_>(scase, u, soph,pmap) {}



                    template <typename T>
                        bool operator()(const T* const C0,const T* const C1,
                                const T* P0, const T* P1, const T* P2, 
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2};
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(3);
                            return this->template call<T,T,T, T>(C0, C1, P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],residuals);
                        }

                    template <typename T>
                        bool operator()(const T* const C0,const T* const C1,
                                const T* P0, const T* P1, const T* P2, const T* P3,
                                T* residuals) const {
                            const T * P[] = {P0,P1,P2,P3};
                            const std::vector<unsigned char> & pmap = this->pmap;
                            this->check_map(4);
                            return this->template call<T,T,T, T>(C0, C1, P[pmap[SPLINE_P0]],P[pmap[SPLINE_P1]],
                                    P[pmap[SPLINE_P2]],P[pmap[SPLINE_P3]],residuals);
                        }

                    template <typename T1, typename T2, typename T>
                        bool call(const T1* const C0, const T1* const C1,
                                const T2* const P0, const T2* const P1, const T2* const P2, const T2* const P3, 
                                T* residuals) const {
                            using Map = Eigen::Map<Eigen::Matrix<T2,kDim,1> const>;
                            Map p0(P0), p1(P1), p2(P2), p3(P3);
                            
                            sophus::details::CubicBSplineSegment<T2,kDim> s(this->segment_case,p0,p1,p2,p3);
                            Vector<T> Tt = s.interpolate(this->u).template cast<T>();
                            return this->soph->operator()(C0,C1,Tt.data(),residuals);

                        }

                };



            //////////////////////////////////////////////////////////////////////////////////////////////
            //
            // Helper functions to insert residual blocks using splines



            // Add an autodiff'ed residual function to a problem defined on a spline. The residual functor is expected to take as 
            // as arguments a class par1, another class par2 and to be estimated at the spline position t on an Vector<Scalar_,kDim> class
            // 
            template <class ParamClass1,class ParamClass2,class ErrorFunctor,int num_residuals>
                static bool addResidualFunction2(::ceres::Problem &problem, 
                        ParamClass1 & par1, ParamClass2 & par2, double t, 
                        std::shared_ptr<BSplined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ::ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineErrorWrapper2<ErrorFunctor,num_residuals>;
                    KnotsAndU ku = spline->knotsAndU(t);
                    std::vector<unsigned char> map(4,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_prev)].push_back(SPLINE_P0);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_0)].push_back(SPLINE_P1);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_1)].push_back(SPLINE_P2);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_2)].push_back(SPLINE_P3);
                    std::vector<double *> parameter_blocks;
                    parameter_blocks.push_back(par1.data());
                    parameter_blocks.push_back(par2.data()); 
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size()-2;
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(ku.segment_case,ku.u, functor,pmap);
                    ::ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 5:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          ParamClass1::num_parameters,ParamClass2::num_parameters,
                                          kDim, kDim, kDim> (ew);
                            break;
                        case 6:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          ParamClass1::num_parameters,ParamClass2::num_parameters,
                                          kDim, kDim, kDim, kDim> (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=5) && (parameter_blocks.size()<=6));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);
                    return true;
                }

            template <class ParamClass1,class ErrorFunctor,int num_residuals>
                static bool addResidualFunction1(::ceres::Problem &problem, ParamClass1 & par1, double t, 
                        std::shared_ptr<BSplined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ::ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineErrorWrapper1<ErrorFunctor,num_residuals>;
                    KnotsAndU ku = spline->knotsAndU(t);
                    std::vector<unsigned char> map(4,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_prev)].push_back(SPLINE_P0);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_0)].push_back(SPLINE_P1);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_1)].push_back(SPLINE_P2);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_2)].push_back(SPLINE_P3);
                    std::vector<double *> parameter_blocks;
                    parameter_blocks.push_back(par1.data());
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size()-1;
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(ku.segment_case,ku.u, functor, map);
                    ::ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 4:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, ParamClass1::num_parameters,
                                          kDim, kDim, kDim> (ew);
                            break;
                        case 5:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, ParamClass1::num_parameters,
                                          kDim, kDim, kDim, kDim> (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=4) && (parameter_blocks.size()<=5));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);
                    return true;
                }

            // Add an autodiff'ed residual function to a problem defined on a spline. 
            // The residual functor is assumed to be defined as 
            //
            // class Functor {
            //  public:
            //      Functor() {}
            //      template <class T>
            //        bool operator()(T const * const P, T* residuals) const {
            //          ...
            //          return true;
            //        }
            // };
            // 
            // The functor will be evaluated at t. 
            // 
            // The following functions are provided:
            // - addResidualFunction0(problem,t,spline,functor,loss_function): 
            //      functor defined with:
            //      template <class T>
            //        bool operator()(T const * const P, T* residuals) const { ... }
            //
            // - addResidualFunction1(problem,par1,t,spline,functor,loss_function): 
            //      functor defined with:
            //      template <class T1, class T2, class T>
            //        bool operator()(T1 const * const C1, T2 const * const C2, 
            //        T const * const P, T* residuals) const { ... }
            //      where C1 is the representation of par1 (par1.data()), which can
            //      can be a parameter or calibration class.
            //
            // - addResidualFunction2(problem,par1,par2,t,spline,functor,loss_function): 
            //      functor defined with:
            //      template <class T1, class T2, class T>
            //        bool operator()(T1 const * const C1, T2 const * const C2, 
            //        T const * const P, T* residuals) const { ... }
            //      where C1 is the representation of par1 (par1.data()), which can
            //      can be a parameter or calibration class, and C2 is similarly
            //      another calibration parameter.
            //
            // - addResidualFunction2Points0(problem,t1,t2,spline,functor,loss_function): 
            //      functor defined with:
            //      template <class T>
            //        bool operator()(T const * const P, 
            //        T const * const Q, T* residuals) const { ... }
            //      where P and Q are two representations, sampled on 
            //      the spline at t1 and t2.
            //
            // - addResidualFunction2Points0(problem,t1,spline1,t2,spline2,functor,loss_function): 
            //      functor defined with:
            //      template <class T>
            //        bool operator()(T const * const P, 
            //        T const * const Q, T* residuals) const { ... }
            //      where P and Q are two lie-group representations, sampled on 
            //      the spline1 at t1 and spline2 at t2.
            //
            //
            template <class ErrorFunctor,int num_residuals>
                static bool addResidualFunction0(::ceres::Problem &problem, double t, 
                        std::shared_ptr<BSplined> spline,
                        std::shared_ptr<ErrorFunctor> functor, 
                        ::ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineErrorWrapper0<ErrorFunctor,num_residuals>;
                    KnotsAndU ku = spline->knotsAndU(t);
                    std::vector<unsigned char> map(4,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_prev)].push_back(SPLINE_P0);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_0)].push_back(SPLINE_P1);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_1)].push_back(SPLINE_P2);
                    pmap[spline->unsafeMutControlPointPtr(ku.idx_2)].push_back(SPLINE_P3);
                    std::vector<double *> parameter_blocks;
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size();
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(ku.segment_case,ku.u, functor, map);
                    ::ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 3:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim> (ew);
                            break;
                        case 4:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim> (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=3) && (parameter_blocks.size()<=4));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);
                    return true;
                }

            template <class ErrorFunctor,int num_residuals>
                static bool addResidualFunction2Points0(::ceres::Problem &problem, 
                        double t1, double t2, 
                        std::shared_ptr<BSplined> spline,
                        std::shared_ptr<ErrorFunctor> functor, ::ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineError2PointsWrapper0<ErrorFunctor,num_residuals>;
                    KnotsAndU ku1 = spline->knotsAndU(t1);
                    KnotsAndU ku2 = spline->knotsAndU(t2);
                    std::vector<unsigned char> map(8,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline->unsafeMutControlPointPtr(ku1.idx_prev)].push_back(SPLINE_P0);
                    pmap[spline->unsafeMutControlPointPtr(ku1.idx_0)].push_back(SPLINE_P1);
                    pmap[spline->unsafeMutControlPointPtr(ku1.idx_1)].push_back(SPLINE_P2);
                    pmap[spline->unsafeMutControlPointPtr(ku1.idx_2)].push_back(SPLINE_P3);
                    pmap[spline->unsafeMutControlPointPtr(ku2.idx_prev)].push_back(SPLINE_Q0);
                    pmap[spline->unsafeMutControlPointPtr(ku2.idx_0)].push_back(SPLINE_Q1);
                    pmap[spline->unsafeMutControlPointPtr(ku2.idx_1)].push_back(SPLINE_Q2);
                    pmap[spline->unsafeMutControlPointPtr(ku2.idx_2)].push_back(SPLINE_Q3);
                    std::vector<double *> parameter_blocks;
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size();
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(ku1.segment_case,ku1.u, ku2.segment_case,ku2.u, functor, pmap);
                    ::ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 3:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim> (ew);
                            break;
                        case 4:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim> (ew);
                            break;
                        case 5:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim,
                                          kDim > (ew);
                            break;
                        case 6:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim,
                                          kDim, kDim > (ew);
                            break;
                        case 7:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim,
                                          kDim, kDim, kDim > (ew);
                            break;
                        case 8:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim,
                                          kDim, kDim, kDim, kDim > (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=3) && (parameter_blocks.size()<=8));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);

                    return true;
                }

            template <class ErrorFunctor,int num_residuals>
                static bool addResidual2Functions0(::ceres::Problem &problem, 
                        unsigned int derivative_order, 
                        double delta_t1, double t1, std::shared_ptr<BSplined> spline1,
                        double delta_t2, double t2, std::shared_ptr<BSplined> spline2,
                        std::shared_ptr<ErrorFunctor> functor, ::ceres::LossFunction * loss_function = nullptr) {
                    using Wrapper = SplineError2FunctionsWrapper0<ErrorFunctor,num_residuals>;
                    KnotsAndU ku1 = spline1->knotsAndU(t1);
                    KnotsAndU ku2 = spline2->knotsAndU(t2);
                    std::vector<unsigned char> map(8,255);
                    std::map<double *,std::vector<unsigned char>> pmap;
                    pmap[spline1->unsafeMutControlPointPtr(ku1.idx_prev)].push_back(SPLINE_P0);
                    pmap[spline1->unsafeMutControlPointPtr(ku1.idx_0)].push_back(SPLINE_P1);
                    pmap[spline1->unsafeMutControlPointPtr(ku1.idx_1)].push_back(SPLINE_P2);
                    pmap[spline1->unsafeMutControlPointPtr(ku1.idx_2)].push_back(SPLINE_P3);
                    pmap[spline2->unsafeMutControlPointPtr(ku2.idx_prev)].push_back(SPLINE_Q0);
                    pmap[spline2->unsafeMutControlPointPtr(ku2.idx_0)].push_back(SPLINE_Q1);
                    pmap[spline2->unsafeMutControlPointPtr(ku2.idx_1)].push_back(SPLINE_Q2);
                    pmap[spline2->unsafeMutControlPointPtr(ku2.idx_2)].push_back(SPLINE_Q3);
                    std::vector<double *> parameter_blocks;
                    for (auto it : pmap) {
                        for (unsigned char x : it.second) {
                            map[x] = parameter_blocks.size();
                        }
                        parameter_blocks.push_back(it.first);
                    }
                    Wrapper * ew = new Wrapper(ku1.segment_case,ku1.u, ku2.segment_case,ku2.u, functor, pmap);
                    ::ceres::CostFunction *cost_function = NULL;
                    switch (parameter_blocks.size()) {
                        case 6:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim,
                                          kDim, kDim > (ew);
                            break;
                        case 7:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim,
                                          kDim, kDim, kDim > (ew);
                            break;
                        case 8:
                            cost_function =  new ::ceres::AutoDiffCostFunction<Wrapper,
                                          num_residuals, 
                                          kDim, kDim, kDim, kDim,
                                          kDim, kDim, kDim, kDim > (ew);
                            break;
                        default:
                            assert((parameter_blocks.size()>=6) && (parameter_blocks.size()<=8));
                    }

                    problem.AddResidualBlock(cost_function, loss_function, parameter_blocks);

                    return true;
                }

        };

}

