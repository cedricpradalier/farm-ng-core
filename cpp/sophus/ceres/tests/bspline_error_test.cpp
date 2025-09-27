// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/calculus/num_diff.h"
#include "sophus/interp/interpolate.h"
#include "sophus/interp/spline/bspline.h"
#include "sophus/interp/spline/group_bspline.h"
#include "sophus/ceres/bspline_error.h"
#include "sophus/ceres/group_bspline_error.h"
#include "sophus/lie/isometry2.h"
#include "sophus/lie/isometry3.h"
#include "sophus/lie/rotation2.h"
#include "sophus/lie/rotation3.h"
#include "sophus/lie/scaling.h"
#include "sophus/lie/scaling_translation.h"
#include "sophus/lie/similarity2.h"
#include "sophus/lie/similarity3.h"
#include "sophus/lie/spiral_similarity2.h"
#include "sophus/lie/spiral_similarity3.h"
#include "sophus/lie/translation.h"

#include <ceres/ceres.h>
#include <sophus/ceres/manifold.h>
#include "ceres_flags.hpp"

#include <gtest/gtest.h>

namespace sophus::test {

template <typename Scalar,int kDim>
    struct BSplinePropTestSuite {
        static int constexpr kDof = kDim;
        static int constexpr kNumParams = kDim;
        using SES = BSplineErrorSupport<Scalar,kDim>;
        using Splined = typename SES::BSplined;
        using Vectord = typename SES::BSplined::Vector;

        struct TestCartesianCostFunctor {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                TestCartesianCostFunctor(const Vectord & kT_aw) : T_aw(kT_aw) {}

            template <class T>
                bool operator()(T const* const sT_wa, T* sResiduals) const {
                    Eigen::Map<Eigen::Matrix<T,kNumParams,1> const> const T_wa(sT_wa);
                    Eigen::Map<Eigen::Matrix<T,kNumParams,1>> residuals(sResiduals);

                    residuals = T_aw - T_wa;
                    return true;
                }

            Vectord T_aw;
        };

        template <int N>
            static double squaredNorm(const Eigen::Vector<double, N>& vec) {
                return vec.squaredNorm();
            }



        static void runAllTests(std::string group_name) {
            auto kElementExamples = ::sophus::pointExamples<Scalar,kDim>();
            if (kElementExamples.size()<10) {
                size_t initial_size = kElementExamples.size();
                while (kElementExamples.size() < 10) {
                    kElementExamples.push_back(kElementExamples[kElementExamples.size() % initial_size]);
                }
            }

            using Functor = TestCartesianCostFunctor;
            size_t n_knots = 3 * kElementExamples.size() / 4;
            // Running Lie group spline approximation
            std::vector<Vectord> control_poses(n_knots,Vectord());
            std::shared_ptr<Splined> spline(new Splined(control_poses, -1.0,
                        float(kElementExamples.size()+2)/(n_knots-1)));

            ::ceres::Problem problem;
            double initial_error = 0.;
            for (size_t i = 0; i < kElementExamples.size(); ++i) {
                double t = i;
                Vectord pred = spline->interpolate(t);
                Vectord err = kElementExamples[i] - pred;
                initial_error += squaredNorm(err);

                std::shared_ptr<Functor> functor(new Functor(kElementExamples[i]));
                SES::template addResidualFunction0<Functor,kDim>(problem,t,spline,functor);
            }

            ::ceres::Solver::Options options;
            CHECK(StringToLinearSolverType(FLAGS_linear_solver,
                        &options.linear_solver_type));
            CHECK(StringToPreconditionerType(FLAGS_preconditioner,
                        &options.preconditioner_type));
            CHECK(StringToSparseLinearAlgebraLibraryType(
                        FLAGS_sparse_linear_algebra_library,
                        &options.sparse_linear_algebra_library_type));
            options.use_nonmonotonic_steps = FLAGS_nonmonotonic_steps;
            CHECK(StringToTrustRegionStrategyType(FLAGS_trust_region_strategy,
                        &options.trust_region_strategy_type));
            CHECK(StringToDoglegType(FLAGS_dogleg, &options.dogleg_type));
            options.use_inner_iterations = FLAGS_inner_iterations;

            options.gradient_tolerance = 1e-8;
            options.function_tolerance = 1e-8;
            options.parameter_tolerance = 1e-8;
            options.minimizer_progress_to_stdout = false;
            options.max_num_iterations = 500;


            ::ceres::Solver::Summary summary;
            ::ceres::Solve(options, &problem, &summary);
            // std::cout << summary.FullReport() << "\n";


            // Computing final error in the estimates
            double final_error = 0.;
            for (size_t i = 0; i < kElementExamples.size(); ++i) {
                double t = i;
                Vectord pred = spline->interpolate(t);
                Vectord err = kElementExamples[i] - pred;
                final_error += squaredNorm(err);
            }


            // Expecting reasonable decrease of both estimates' errors and residuals
            SOPHUS_ASSERT_WITHIN_REL(
                    summary.final_cost,
                    summary.initial_cost,
                    0.1,
                    "Spline approximation: {}",
                    kDim);
        }
    };

template <typename Scalar,template <typename> class Group_>
    struct GroupBSplinePropTestSuite {
        using Group = Group_<Scalar>;
        using SES = GroupBSplineErrorSupport<Scalar,Group_>;
        using Splined = typename SES::Splined;

        static int constexpr kDof = Group::kDof;
        static int constexpr kNumParams = Group::kNumParams;


        struct TestLieGroupCostFunctor {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

                TestLieGroupCostFunctor(const Group& T_aw) : T_aw(T_aw) {}

            template <class T>
                bool operator()(T const* const sT_wa, T* sResiduals) const {
                    Eigen::Map<Eigen::Matrix<T,kNumParams,1>const> pT_wa(sT_wa);
                    Group_<T> T_wa = Group_<T>::fromParams(pT_wa);
                    // Mapper class is only used to facciliate difference between
                    // SO2 (which uses Scalar as tangent vector type) and other groups
                    // (which use Vector<...> as tangent vector type).
                    //
                    // Feel free to use direct dereferencing or Eigen::Map depending
                    // on ypur use-case for concrete application
                    //
                    // We only use Mapper class in order to make tests universally
                    // compatible with LieGroup::Tangent being Scalar or Vector
                    using Mapper = Eigen::Map<typename Group_<T>::Tangent>;
                    typename Mapper::Map residuals(sResiduals);

                    // We are able to mix Sophus types with doubles and Jet types without
                    // needing to cast to T.
                    residuals = (T_aw.inverse() * T_wa).log();
                    return true;
                }

            Group T_aw;
        };

        template <int N>
            static double squaredNorm(const Eigen::Vector<double, N>& vec) {
                return vec.squaredNorm();
            }



        class NormalizeCallback: public ::ceres::IterationCallback { 
            protected:
                std::shared_ptr<Splined> spline;
            public: 
                NormalizeCallback(std::shared_ptr<Splined> spline) : spline(spline) {}
                void normalize() {
                    // for (size_t i=0;i<spline->parentFromsControlPoint().size();i++) {
                    //     spline->parentFromsControlPoint()[i].normalize();
                    // }
                }
                virtual ::ceres::CallbackReturnType operator()(const 
                        ::ceres::IterationSummary& /*summary*/) { 
                    normalize();
                    return ::ceres::SOLVER_CONTINUE;
                } 
        };



        static void runAllTests(std::string group_name) {
            static int constexpr kDof = Group::kDof;
            std::vector<Group> kElementExamples(10);
            auto kPointExamples = ::sophus::pointExamples<Scalar,4>();
            for (size_t i=0;i<kElementExamples.size();i++) {
                Eigen::Matrix<Scalar,Group::kDof,1> glog = Eigen::Matrix<Scalar,Group::kDof,1>::Random()*1e-3;
                kElementExamples[i] = Group::exp(glog) * Group();
            }
            using Functor = TestLieGroupCostFunctor;
            size_t n_knots = 3 * kElementExamples.size() / 4;
            // Running Lie group spline approximation
            std::vector<Group> control_poses(n_knots,Group());
            std::shared_ptr<Splined> spline(new Splined(control_poses, -1.0,
                        float(kElementExamples.size()+2)/(n_knots-1)));
            NormalizeCallback norm_cb(spline);
            norm_cb.normalize();

            ::ceres::Problem problem;
            double initial_error = 0.;
            auto parametrization = new sophus::ceres::Manifold<Group_>;

            for (auto v : spline->parentFromsControlPoint()) {

                problem.AddParameterBlock(v.unsafeMutPtr(), Group::kNumParams, parametrization);
            }

            for (size_t i = 0; i < kElementExamples.size(); ++i) {
                double t = i;
                Group pred = spline->parentFromSpline(t);
                Group err = kElementExamples[i].inverse() * pred;
                initial_error += squaredNorm(err.log());

                std::shared_ptr<Functor> functor(new Functor(kElementExamples[i]));
                SES::template addResidualFunction0<Functor,kDof>(problem,t,spline,functor);
            }

            ::ceres::Solver::Options options;
            CHECK(StringToLinearSolverType(FLAGS_linear_solver,
                        &options.linear_solver_type));
            CHECK(StringToPreconditionerType(FLAGS_preconditioner,
                        &options.preconditioner_type));
            CHECK(StringToSparseLinearAlgebraLibraryType(
                        FLAGS_sparse_linear_algebra_library,
                        &options.sparse_linear_algebra_library_type));
            options.use_nonmonotonic_steps = FLAGS_nonmonotonic_steps;
            CHECK(StringToTrustRegionStrategyType(FLAGS_trust_region_strategy,
                        &options.trust_region_strategy_type));
            CHECK(StringToDoglegType(FLAGS_dogleg, &options.dogleg_type));
            options.use_inner_iterations = FLAGS_inner_iterations;


            options.update_state_every_iteration = true; 
            options.callbacks.push_back(&norm_cb);
            options.gradient_tolerance = 1e-8;
            options.function_tolerance = 1e-8;
            options.parameter_tolerance = 1e-8;
            options.minimizer_progress_to_stdout = false;
            options.max_num_iterations = 500;


            ::ceres::Solver::Summary summary;
            ::ceres::Solve(options, &problem, &summary);
            // std::cout << summary.FullReport() << "\n";


            // Computing final error in the estimates
            double final_error = 0.;
            for (size_t i = 0; i < kElementExamples.size(); ++i) {
                double t = i;
                Group pred = spline->parentFromSpline(t);
                Group err = kElementExamples[i].inverse() * pred;
                final_error += squaredNorm(err.log());
            }


            // Expecting reasonable decrease of both estimates' errors and residuals
            SOPHUS_ASSERT_WITHIN_REL(
                    summary.final_cost,
                    summary.initial_cost,
                    0.1,
                    "Spline approximation: {}",
                    group_name);
        }
    };

  template <typename T>
      using Translation1 = Translation<T,1>;

TEST(lie_group_bspline, lie_group_bspline_prop_test) {
  // GroupBSplinePropTestSuite<double,Scaling2>::runAllTests("Scaling2");
  // GroupBSplinePropTestSuite<double,Scaling3>::runAllTests("Scaling3");

  // GroupBSplinePropTestSuite<double,Translation1>::runAllTests("Translation1");

  // GroupBSplinePropTestSuite<double,Translation2>::runAllTests("Translation2");
  // GroupBSplinePropTestSuite<double,Translation3>::runAllTests("Translation3");
  // GroupBSplinePropTestSuite<double,ScalingTranslation2>::runAllTests( "ScalingTranslation2");
  // GroupBSplinePropTestSuite<double,ScalingTranslation3>::runAllTests( "ScalingTranslation3");

  GroupBSplinePropTestSuite<double,Rotation2>::runAllTests("Rotation2");
  GroupBSplinePropTestSuite<double,Rotation3>::runAllTests("Rotation3");
  GroupBSplinePropTestSuite<double,Isometry2>::runAllTests("Isometry2");
  GroupBSplinePropTestSuite<double,Isometry3>::runAllTests("Isometry3");

  // GroupBSplinePropTestSuite<double,SpiralSimilarity2>::runAllTests( "SpiralSimilarity2");
  // GroupBSplinePropTestSuite<double,SpiralSimilarity3>::runAllTests( "SpiralSimilarity3");
  // GroupBSplinePropTestSuite<double,Similarity2>::runAllTests("Similarity2");
  // GroupBSplinePropTestSuite<double,Similarity3>::runAllTests("Similarity3");
}

TEST(cartesian_bspline, cartesian_bspline_prop_test) {
  BSplinePropTestSuite<double,1>::runAllTests("Vector1d");
  BSplinePropTestSuite<double,2>::runAllTests("Vector1d");
  BSplinePropTestSuite<double,3>::runAllTests("Vector1d");
  BSplinePropTestSuite<double,4>::runAllTests("Vector1d");
}

}  // namespace sophus::test
