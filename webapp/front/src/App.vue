<template lang="pug">
  v-app(dark)
    v-toolbar(app absolute clipped-left)
      v-toolbar-side-icon(@click.stop="gui.nav_drawer_visible = !gui.nav_drawer_visible")
      v-toolbar-title Style Transfer
    v-navigation-drawer(v-model="gui.nav_drawer_visible" :temporary="gui.nav_drawer_temporary" clipped absolute overflow app)
      v-list
        v-list-tile
          v-select(:items="image_sizes"
                  label="Image Size"
                  v-model="image_size")
        v-divider
        v-list-tile
          v-switch(v-model="gui.fullscreen_image" label="Fullscreen" primary)
    v-content
      v-container(fluid) 
        v-layout(justify-space-around align-space-around)
          v-flex(xs5 v-if="!gui.fullscreen_image")
            v-img(src="/video_feed" aspect-ratio="1.7")
          v-flex(:xs5="!gui.fullscreen_image" :xs10="gui.fullscreen_image") 
            v-img(src="/inference_feed" aspect-ratio="1.7")
    v-footer(app) 
      span(class="px-3") &copy; Sébastien IOOSS {{ new Date().getFullYear() }}
</template>

<script>
const axios = require("axios");

export default {
  name: "App",
  data() {
    return {
      gui: {
        nav_drawer_visible: false,
        fullscreen_image: false
      },
      image_sizes: ["640x480", "1920x1080", "1280x720", "1024x768"],
      image_size: "640x480",
      possible_camera_ports: null
    };
  },
  mounted: function() {
    axios.get("/config").then(function(response) {
      self.possible_camera_ports = response["data"].camera_ports;
    });
  }
};
</script>
