tippy('p,h1,h2,h3,h4,strong,li,em,span', {
  content: 'Loading...',
  //theme: 'white',
  trigger: 'click',
  async onShow(tip) {

    if (!tip.state.ajax) {
      tip.state.ajax = {
        isFetching: false,
        canFetch: true,
      }
    }

    if (tip.state.ajax.isFetching || !tip.state.ajax.canFetch) {
      return
    }

    tip.state.ajax.isFetching = true
    tip.state.ajax.canFetch = false

    try {
      
      var content = tip.reference.innerText;
      var url = "https://translate-blog-luisfredgs.herokuapp.com";
      var params = "src_lg=pt&dest_lg=en&query=" + content;
      var http = new XMLHttpRequest();

      http.open("GET", url+"?"+params, true);
      http.onreadystatechange = function()
      {
          if(http.readyState == 4 && http.status == 200) {
              console.log(http.responseText);

              if (tip.state.isVisible) 
              {
                text = JSON.parse(http.responseText);
                tip.setContent(text.response);
                
              }

          }
      }
      http.send(null);

    } catch (e) {
      tip.setContent(`Fetch failed. ${e}`)
    } finally {
      tip.state.ajax.isFetching = false
    }
  },
  onHidden(tip) {
    tip.state.ajax.canFetch = true
    tip.setContent(INITIAL_CONTENT)
  },
})