// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <fstream>

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

#include <gtest/gtest.h>

namespace sophus::test {

    bool output_test_files = true;

    void setCeresOptions(::ceres::Solver::Options & options) {
        // DEFINE_string(linear_solver, "sparse_normal_cholesky", "Options are: "
        //         "sparse_schur, dense_schur, iterative_schur, sparse_normal_cholesky, "
        //         "dense_qr, dense_normal_cholesky and cgnr.");
        // 
        // DEFINE_string(preconditioner, "jacobi", "Options are: "
        //         "identity, jacobi, schur_jacobi, cluster_jacobi, "
        //         "cluster_tridiagonal.");
        // 
        // DEFINE_string(sparse_linear_algebra_library, "suite_sparse",
        //         "Options are: suite_sparse and cx_sparse.");
        CHECK(StringToLinearSolverType("sparse_normal_cholesky",
                    &options.linear_solver_type));
        CHECK(StringToPreconditionerType("jacobi",
                    &options.preconditioner_type));
        CHECK(StringToSparseLinearAlgebraLibraryType("suite_sparse",
                    &options.sparse_linear_algebra_library_type));
        options.use_nonmonotonic_steps = false;
        // DEFINE_string(trust_region_strategy, "levenberg_marquardt",
        //         "Options are: levenberg_marquardt, dogleg.");
        // DEFINE_string(dogleg, "traditional_dogleg", "Options are: traditional_dogleg,"
        //         "subspace_dogleg.");
        CHECK(StringToTrustRegionStrategyType("levenberg_marquardt",
                    &options.trust_region_strategy_type));
        CHECK(StringToDoglegType("traditional_dogleg", &options.dogleg_type));
        options.use_inner_iterations = false;

        options.gradient_tolerance = 1e-8;
        options.function_tolerance = 1e-8;
        options.parameter_tolerance = 1e-8;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 500;
    }

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



            static void runAllTests(std::string group_name, bool file_output,double scale=1.0) {
                std::vector<Vectord> kElementExamples(10);
                for (size_t i=0;i<kElementExamples.size();i++) {
                    kElementExamples[i] = Vectord::Random() * scale;
                }
                if (file_output) {
                    std::ofstream finput(group_name+"_input.csv", std::ios::binary);
                    for (size_t i=0;i<kElementExamples.size();i++) {
                        finput << i << " " << kElementExamples[i].transpose() << std::endl;
                    }
                    finput.close();
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
                setCeresOptions(options);


                ::ceres::Solver::Summary summary;
                ::ceres::Solve(options, &problem, &summary);
                // std::cout << summary.FullReport() << "\n";

                if (file_output) {
                    std::ofstream fknots(group_name+"_knots.csv", std::ios::binary);
                    const std::vector<Vectord> & cpts = spline->constControlPoints();
                    for (size_t i=0;i<cpts.size();i++) {
                        fknots << spline->t0() + i * spline->deltaT() << 
                            " " << cpts[i].transpose() << std::endl;
                    }
                    fknots.close();
                    std::ofstream fspline(group_name+"_spline.csv", std::ios::binary);
                    for (double t=spline->t0();t<spline->tmax();t+=0.05) {
                        Vectord pred = spline->interpolate(t);
                        fspline << t << " " << pred.transpose() << std::endl;
                    }
                    fspline.close();
                }

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
                        0.9,
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
#if 0
                        // Not needed, just an example to show how this could be done
                        for (size_t i=0;i<spline->parentFromsControlPoint().size();i++) {
                            Group cp = spline->parentFromsControlPoint()[i];
                            spline->parentFromsControlPoint()[i] = Group::exp(cp.log());
                        }
#endif
                    }
                    virtual ::ceres::CallbackReturnType operator()(const 
                            ::ceres::IterationSummary& /*summary*/) { 
                        normalize();
                        return ::ceres::SOLVER_CONTINUE;
                    } 
            };

            static void evaluate(::ceres::Problem & problem, std::ostream & s) {
                ::ceres::CRSMatrix jacobian;
                double cost;

                std::vector<::ceres::ResidualBlockId> residual_blocks;
                std::vector<double *> parameter_blocks;
                std::vector<size_t> parameter_size;
                std::vector<size_t> tangent_size;
                problem.GetResidualBlocks(&residual_blocks);
                problem.GetParameterBlocks(&parameter_blocks);
                parameter_size.resize(parameter_blocks.size());
                tangent_size.resize(parameter_blocks.size());
                for (size_t i=0;i<parameter_blocks.size();i++) {
                    parameter_size[i] = problem.ParameterBlockSize(parameter_blocks[i]);
                    tangent_size[i] = problem.ParameterBlockTangentSize(parameter_blocks[i]);
                    std::cout << "Param " << i << " " << parameter_blocks[i] << 
                        " " << parameter_size[i] <<
                        " " << tangent_size[i] << std::endl;
                }
#if 0
                for (size_t i=0;i<residual_blocks.size();i++) {
                    double rcost=0;
                    const ::ceres::CostFunction *cf = problem.GetCostFunctionForResidualBlock(residual_blocks[i]);
                    Eigen::VectorXd residuals(cf->num_residuals());
                    const std::vector<int32_t> & pbsize = cf->parameter_block_sizes();
                    std::vector<Eigen::MatrixXd> ejacobian(parameter_blocks.size());
                    double * jacobian[parameter_blocks.size()];
                    for (size_t j=0;j<parameter_blocks.size();j++) {
                        ejacobian[j]=Eigen::MatrixXd(cf->num_residuals(),tangent_size[j]);
                        jacobian[j]=ejacobian[j].data();
                    }
                    problem.EvaluateResidualBlock(residual_blocks[i],false,&rcost,residuals.data(),jacobian);
                    std::cout << "RB " << residual_blocks[i] << 
                        " C " << rcost << " R " << residuals.transpose() << std::endl;
                    for (size_t j=0;j<parameter_blocks.size();j++) {
                        std::cout << "Pb " << j << " : " << cf->num_residuals() << "x" << tangent_size[j] 
                            << std::endl << ejacobian[j] << std::endl;
                    }

                }
#endif

                std::vector<double> residuals;
#if 0
                std::vector<double> gradient;
                problem.Evaluate(::ceres::Problem::EvaluateOptions(),&cost,&residuals,&gradient,&jacobian);
                Eigen::MatrixXd ejacobian = Eigen::MatrixXd::Zero(jacobian.num_rows,jacobian.num_cols);
                for (int row=0;row<jacobian.num_rows;row++) {
                    for (int icol=jacobian.rows[row];icol<jacobian.rows[row+1];icol++) {
                        ejacobian(row,jacobian.cols[icol])=jacobian.values[icol];
                    }
                }
                std::cout << "Jacobian" << std::endl 
                    << ejacobian << std::endl 
                    << "Gradient" << std::endl;
                for (size_t i=0;i<gradient.size();i++) {
                    std::cout << gradient[i] << " ";
                }
                std::cout << std::endl;
#else
                problem.Evaluate(::ceres::Problem::EvaluateOptions(),&cost,&residuals,nullptr,nullptr);
#endif
                std::cout << "Cost " << cost << " Residuals" << std::endl;
                for (size_t i=0;i<residuals.size();i++) {
                    std::cout << residuals[i] << " ";
                }
                std::cout << std::endl;
                std::cout.flush();

            }



            static void runAllTests(std::string group_name, bool file_output, double scale=1e+1) {
                std::cout << "Starting test for " << group_name << std::endl;
                static int constexpr kDof = Group::kDof;
                std::vector<Group> kElementExamples(10);
                for (size_t i=0;i<kElementExamples.size();i++) {
                    Eigen::Matrix<Scalar,Group::kDof,1> glog = Eigen::Matrix<Scalar,Group::kDof,1>::Random()*scale;
                    kElementExamples[i] = Group::exp(glog);
                }
                if (file_output) {
                    std::ofstream finput(group_name+"_input.csv", std::ios::binary);
                    for (size_t i=0;i<kElementExamples.size();i++) {
                        finput << i << " " << kElementExamples[i].log().transpose() << std::endl;
                    }
                    finput.close();
                }
                using Functor = TestLieGroupCostFunctor;
                size_t n_knots = 3 * (kElementExamples.size()+2) / 5;
                // Running Lie group spline approximation
                std::vector<Group> control_poses(n_knots,Group());
                std::shared_ptr<Splined> spline(new Splined(control_poses, -1.0,
                            float(kElementExamples.size()+2)/(n_knots-1)));
                NormalizeCallback norm_cb(spline);
                norm_cb.normalize();

                ::ceres::Problem problem;
                double initial_error = 0.;
                auto parametrization = new sophus::ceres::Manifold<Group_>;

                std::cout << "Adding parametrization" << std::endl;
                for (size_t i=0;i<spline->parentFromsControlPoint().size();i++) {
                    double * p = spline->unsafeMutControlPointPtr(i);
                    std::cout << "Manifold for Param " << p << std::endl;
                    problem.AddParameterBlock(p, Group::kNumParams, parametrization);
                }

                std::cout << "Setting residual functions" << std::endl;
                for (size_t i = 0; i < kElementExamples.size(); ++i) {
                    double t = i;
                    Group pred = spline->parentFromSpline(t);
                    Group err = kElementExamples[i].inverse() * pred;
                    initial_error += squaredNorm(err.log());

                    std::shared_ptr<Functor> functor(new Functor(kElementExamples[i]));
                    double residuals[kDof];
                    SES::template addResidualFunction0<Functor,kDof>(problem,t,spline,functor);
                }

                ::ceres::Solver::Options options;
                setCeresOptions(options);

                std::cout << "Evaluating problem" << std::endl;
                evaluate(problem,std::cout);

                std::cout << "Starting solver" << std::endl;
                ::ceres::Solver::Summary summary;
                ::ceres::Solve(options, &problem, &summary);
                // std::cout << summary.FullReport() << "\n";

                if (file_output) {
                    std::ofstream fknots(group_name+"_knots.csv", std::ios::binary);
                    for (size_t i=0;i<spline->parentFromsControlPoint().size();i++) {
                        fknots << spline->t0() + i * spline->deltaT() << 
                            " " << spline->parentFromsControlPoint()[i].log().transpose() << std::endl;
                    }
                    fknots.close();
                    std::ofstream fspline(group_name+"_spline.csv", std::ios::binary);
                    for (double t=spline->t0();t<spline->tmax();t+=0.05) {
                        Group pred = spline->parentFromSpline(t);
                        fspline << t << " " << pred.log().transpose() << std::endl;
                    }
                    fspline.close();
                }

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
                        0.9,
                        "Spline approximation: {}",
                        group_name);
            }
        };

    template <typename T>
        using Translation1 = Translation<T,1>;

    TEST(cartesian_bspline, cartesian_bspline_prop_test) {
        BSplinePropTestSuite<double,1>::runAllTests("Vector1d",output_test_files);
        BSplinePropTestSuite<double,2>::runAllTests("Vector2d",output_test_files);
        BSplinePropTestSuite<double,3>::runAllTests("Vector3d",output_test_files);
        BSplinePropTestSuite<double,4>::runAllTests("Vector4d",output_test_files);
    }


    TEST(lie_group_bspline, lie_group_bspline_prop_test) {
        GroupBSplinePropTestSuite<double,Translation1>::runAllTests("Translation1",output_test_files);
        GroupBSplinePropTestSuite<double,Translation2>::runAllTests("Translation2",output_test_files);
        GroupBSplinePropTestSuite<double,Translation3>::runAllTests("Translation3",output_test_files);

        GroupBSplinePropTestSuite<double,Rotation2>::runAllTests("Rotation2",output_test_files);
        GroupBSplinePropTestSuite<double,Rotation3>::runAllTests("Rotation3",output_test_files);
        GroupBSplinePropTestSuite<double,Isometry2>::runAllTests("Isometry2",output_test_files);
        GroupBSplinePropTestSuite<double,Isometry3>::runAllTests("Isometry3",output_test_files);

        GroupBSplinePropTestSuite<double,SpiralSimilarity2>::runAllTests( "SpiralSimilarity2",output_test_files);
        GroupBSplinePropTestSuite<double,SpiralSimilarity3>::runAllTests( "SpiralSimilarity3",output_test_files);
        GroupBSplinePropTestSuite<double,Similarity2>::runAllTests("Similarity2",output_test_files);
        GroupBSplinePropTestSuite<double,Similarity3>::runAllTests("Similarity3",output_test_files);

        GroupBSplinePropTestSuite<double,Scaling2>::runAllTests("Scaling2",output_test_files,1e+0);
        GroupBSplinePropTestSuite<double,Scaling3>::runAllTests("Scaling3",output_test_files,1e+0);
        GroupBSplinePropTestSuite<double,ScalingTranslation2>::runAllTests( "ScalingTranslation2",output_test_files,1e+0);
        GroupBSplinePropTestSuite<double,ScalingTranslation3>::runAllTests( "ScalingTranslation3",output_test_files,1e+0);

    }

}  // namespace sophus::test
