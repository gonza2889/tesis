"use strict";(self.webpackChunkmermaid_viewer=self.webpackChunkmermaid_viewer||[]).push([[679],{1679:function(e,t,a){a.r(t),a.d(t,{diagram:function(){return I}});var r=a(4165),n=a(7762),o=a(5861),s=a(8003),c=a(8433),i=a(8225),d=a(366),l=a(4726),p=(a(7892),a(504),a(8703),a(1818),a(7351),"rect"),u="rectWithTitle",g="statediagram",h="".concat(g,"-").concat("state"),b="transition",y="".concat(b," ").concat("note-edge"),f="".concat(g,"-").concat("note"),v="".concat(g,"-").concat("cluster"),w="".concat(g,"-").concat("cluster-alt"),m="parent",x="note",T="----",k="".concat(T).concat(x),S="".concat(T).concat(m),D="fill:none",A="fill: #333",B="text",E="normal",C={},R=0;function V(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:"",t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:0,a=arguments.length>2&&void 0!==arguments[2]?arguments[2]:"",r=arguments.length>3&&void 0!==arguments[3]?arguments[3]:T,n=null!==a&&a.length>0?"".concat(r).concat(a):"";return"".concat("state","-").concat(e).concat(n,"-").concat(t)}var N=function(e,t,a,r,n,o){var c,i=a.id,l=void 0===(c=r[i])||null===c?"":c.classes?c.classes.join(" "):"";if("root"!==i){var g=p;!0===a.start&&(g="start"),!1===a.start&&(g="end"),a.type!==s.D&&(g=a.type),C[i]||(C[i]={id:i,shape:g,description:d.e.sanitizeText(i,(0,d.c)()),classes:"".concat(l," ").concat(h)});var b=C[i];a.description&&(Array.isArray(b.description)?(b.shape=u,b.description.push(a.description)):b.description.length>0?(b.shape=u,b.description===i?b.description=[a.description]:b.description=[b.description,a.description]):(b.shape=p,b.description=a.description),b.description=d.e.sanitizeTextOrArray(b.description,(0,d.c)())),1===b.description.length&&b.shape===u&&(b.shape=p),!b.type&&a.doc&&(d.l.info("Setting cluster for ",i,M(a)),b.type="group",b.dir=M(a),b.shape=a.type===s.a?"divider":"roundedWithTitle",b.classes=b.classes+" "+v+" "+(o?w:""));var T={labelStyle:"",shape:b.shape,labelText:b.description,classes:b.classes,style:"",id:i,dir:b.dir,domId:V(i,R),type:b.type,padding:15,centerLabel:!0};if(a.note){var N={labelStyle:"",shape:"note",labelText:a.note.text,classes:f,style:"",id:i+k+"-"+R,domId:V(i,R,x),type:b.type,padding:15},Z={labelStyle:"",shape:"noteGroup",labelText:a.note.text,classes:b.classes,style:"",id:i+S,domId:V(i,R,m),type:"group",padding:0};R++;var z=i+S;e.setNode(z,Z),e.setNode(N.id,N),e.setNode(i,T),e.setParent(i,z),e.setParent(N.id,z);var I=i,P=N.id;"left of"===a.note.position&&(I=N.id,P=i),e.setEdge(I,P,{arrowhead:"none",arrowType:"",style:D,labelStyle:"",classes:y,arrowheadStyle:A,labelpos:"c",labelType:B,thickness:E})}else e.setNode(i,T)}t&&"root"!==t.id&&(d.l.trace("Setting node ",i," to be child of its parent ",t.id),e.setParent(i,t.id)),a.doc&&(d.l.trace("Adding nodes children "),L(e,a,a.doc,r,n,!o))},L=function(e,t,a,r,n,o){d.l.trace("items",a),a.forEach((function(a){switch(a.stmt){case s.b:case s.D:N(e,t,a,r,n,o);break;case s.S:N(e,t,a.state1,r,n,o),N(e,t,a.state2,r,n,o);var c={id:"edge"+R,arrowhead:"normal",arrowTypeEnd:"arrow_barb",style:D,labelStyle:"",label:d.e.sanitizeText(a.description,(0,d.c)()),arrowheadStyle:A,labelpos:"c",labelType:B,thickness:E,classes:b};e.setEdge(a.state1.id,a.state2.id,c,R),R++}}))},M=function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:s.c;if(e.doc)for(var a=0;a<e.doc.length;a++){var r=e.doc[a];"dir"===r.stmt&&(t=r.value)}return t},Z=function(){var e=(0,o.Z)((0,r.Z)().mark((function e(t,a,o,s){var u,h,b,y,f,v,w,m,x,T,k,S,D,A,B,E,R,V,L,Z,z,I;return(0,r.Z)().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return d.l.info("Drawing state diagram (v2)",a),C={},s.db.getDirection(),u=(0,d.c)(),h=u.securityLevel,b=u.state,y=b.nodeSpacing||50,f=b.rankSpacing||50,d.l.info(s.db.getRootDocV2()),s.db.extract(s.db.getRootDocV2()),d.l.info(s.db.getRootDocV2()),v=s.db.getStates(),w=new c.k({multigraph:!0,compound:!0}).setGraph({rankdir:M(s.db.getRootDocV2()),nodesep:y,ranksep:f,marginx:8,marginy:8}).setDefaultEdgeLabel((function(){return{}})),N(w,void 0,s.db.getRootDocV2(),v,s.db,!0),"sandbox"===h&&(m=(0,i.Ys)("#i"+a)),x="sandbox"===h?(0,i.Ys)(m.nodes()[0].contentDocument.body):(0,i.Ys)("body"),T=x.select('[id="'.concat(a,'"]')),k=x.select("#"+a+" g"),e.next=18,(0,l.r)(k,w,["barb"],g,a);case 18:8,d.u.insertTitle(T,"statediagramTitleText",b.titleTopMargin,s.db.getDiagramTitle()),S=T.node().getBBox(),D=S.width+16,A=S.height+16,T.attr("class",g),B=T.node().getBBox(),(0,d.i)(T,A,D,b.useMaxWidth),E="".concat(B.x-8," ").concat(B.y-8," ").concat(D," ").concat(A),d.l.debug("viewBox ".concat(E)),T.attr("viewBox",E),R=document.querySelectorAll('[id="'+a+'"] .edgeLabel .label'),V=(0,n.Z)(R);try{for(V.s();!(L=V.n()).done;)Z=L.value,z=Z.getBBox(),(I=document.createElementNS("http://www.w3.org/2000/svg",p)).setAttribute("rx",0),I.setAttribute("ry",0),I.setAttribute("width",z.width),I.setAttribute("height",z.height),Z.insertBefore(I,Z.firstChild)}catch(t){V.e(t)}finally{V.f()}case 32:case"end":return e.stop()}}),e)})));return function(t,a,r,n){return e.apply(this,arguments)}}(),z={setConf:function(e){for(var t=0,a=Object.keys(e);t<a.length;t++){e[a[t]]}},getClasses:function(e,t){d.l.trace("Extracting classes"),t.db.clear();try{return t.parser.parse(e),t.db.extract(t.db.getRootDocV2()),t.db.getClasses()}catch(a){return a}},draw:Z},I={parser:s.p,db:s.d,renderer:z,styles:s.s,init:function(e){e.state||(e.state={}),e.state.arrowMarkerAbsolute=e.arrowMarkerAbsolute,s.d.clear()}}}}]);
//# sourceMappingURL=679.d92c2751.chunk.js.map