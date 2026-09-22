plugins {
    id("com.android.application")
    id("kotlin-android")

    // Flutter plugin must be after
    // Android + Kotlin plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.air_control"

    compileSdk =
        flutter.compileSdkVersion

    ndkVersion =
        flutter.ndkVersion


    // ========================================================
    // JAVA
    // ========================================================

    compileOptions {
        sourceCompatibility =
            JavaVersion.VERSION_17

        targetCompatibility =
            JavaVersion.VERSION_17
    }


    // ========================================================
    // KOTLIN
    // ========================================================

    kotlinOptions {
        jvmTarget =
            JavaVersion.VERSION_17
                .toString()
    }


    // ========================================================
    // APP CONFIGURATION
    // ========================================================

    defaultConfig {
        applicationId =
            "com.example.air_control"

        // tflite_flutter Android support is safest
        // from API 26 upward.
        minSdk = 26

        targetSdk =
            flutter.targetSdkVersion

        versionCode =
            flutter.versionCode

        versionName =
            flutter.versionName
    }


    // ========================================================
    // BUILD TYPES
    // ========================================================

    buildTypes {
        release {
            // Development project:
            // use debug signing for now.
            signingConfig =
                signingConfigs
                    .getByName("debug")
        }
    }
}   


flutter {
    source = "../.."
}