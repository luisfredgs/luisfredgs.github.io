INITIAL_CONTENT="Loading..."
tippy('p,h1,h2,h3,h4,strong,li,em,span',{content:'Loading...',trigger:'click',async onShow(tip){if(!tip.state.ajax){tip.state.ajax={isFetching:!1,canFetch:!0,}}
if(tip.state.ajax.isFetching||!tip.state.ajax.canFetch){return}
tip.state.ajax.isFetching=!0
tip.state.ajax.canFetch=!1
try{var content=tip.reference.innerText;var url="https://translate-blog-luisfredgs.herokuapp.com";var params="src_lg=pt&dest_lg=en&query="+content;var http=new XMLHttpRequest();http.open("GET",url+"?"+params,!0);http.onreadystatechange=function()
{if(http.readyState==4&&http.status==200){console.log(http.responseText);if(tip.state.isVisible)
{text=JSON.parse(http.responseText);tip.setContent(text.response)}}}
http.send(null)}catch(e){tip.setContent(`Fetch failed. ${e}`)}finally{tip.state.ajax.isFetching=!1}},onHidden(tip){tip.state.ajax.canFetch=!0
tip.setContent(INITIAL_CONTENT)},})