#[[
sophus_ceres



]]
if(${BUILD_SOPHUS_TESTS})
    message("Building sophus-ceres integration tests")
    find_package(Ceres 2.1.0 REQUIRED)
    add_compile_options(-g -ggdb -O0)
    foreach(test_basename
      ceres
      manifold
      bspline_error
    )
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_ceres
                          LINK_LIBRARIES
                          Ceres::ceres sophus_ceres 
                          #farm_ng_core::farm_ng_core_pipeline
                          sophus_sensor
                          LABELS large)
    endforeach()
else()
    message("NOT Building sophus-ceres integration tests")
endif()
