// 1. open your (google-chrome or another browser) browser.
// 2. press 'ctr + shift + i' to open browser console (without single quote).
// 3. copy-paste 'images?q=' to filter box (without single quote).

// 4. copy - paste line 6 to 8 to your browser (google-chrome) console, and press enter.
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);

// 5. copy - paste line 11 to 17 to your browser (google-chrome) console, and press enter.
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });
var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'urls.txt';
hiddenElement.click();

// 6. download urls as txt file (urls.txt)