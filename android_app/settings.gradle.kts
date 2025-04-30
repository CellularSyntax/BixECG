pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        maven { url = uri("https://jitpack.io") } // Needed for MPAndroidChart
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.PREFER_SETTINGS) // <- Important fix here!
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") } // Needed for MPAndroidChart
    }
}

rootProject.name = "MockECGViewer"
include(":app")
